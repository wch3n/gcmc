"""CHE post-processing for site-conditioned OER states."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from ase.build import molecule
from ase.geometry import get_distances
from ase.io import iread, read
from ase.thermochemistry import IdealGasThermo

from gcmc.utils import build_surface_site_registry

from .output_files import output_path
from .parent_sites import write_csv


_DEFAULT_REFERENCE_THERMO = {
    "enabled": False,
    "temperature_K": None,
    "pressure_Pa": 101325.0,
    "atoms": None,
    "geometry": None,
    "symmetrynumber": None,
    "spin": None,
    "vib_energies_eV": [],
    "ignore_imag_modes": False,
}

_DEFAULT_CHE_CONFIG = {
    "enabled": False,
    "clean_state": "00_clean",
    "clean_reference_mode": "per_site",
    "oh_state": "01_OH",
    "o_state": "02_O",
    "ooh_state": "03_OOH",
    "h2_energy_eV": None,
    "h2o_energy_eV": None,
    "o2_energy_eV": None,
    "h2_correction_eV": 0.0,
    "h2o_correction_eV": 0.0,
    "o2_correction_eV": 0.0,
    "clean_correction_eV": 0.0,
    "oh_correction_eV": 0.0,
    "o_correction_eV": 0.0,
    "ooh_correction_eV": 0.0,
    "oer_reference_mode": "auto",
    "total_oer_free_energy_eV": 4.92,
    "equilibrium_potential_V": 1.23,
    "potential_V": 0.0,
    "use_converged_only": True,
    "allow_unconverged_fallback": False,
    "use_vibrational_free_energies": True,
    "vibration_summary_csv": None,
    "h2_thermo": {
        **_DEFAULT_REFERENCE_THERMO,
        "geometry": "linear",
        "symmetrynumber": 2,
        "spin": 0.0,
    },
    "h2o_thermo": {
        **_DEFAULT_REFERENCE_THERMO,
        "geometry": "nonlinear",
        "symmetrynumber": 2,
        "spin": 0.0,
    },
    "o2_thermo": {
        **_DEFAULT_REFERENCE_THERMO,
        "geometry": "linear",
        "symmetrynumber": 2,
        "spin": 1.0,
    },
    "route_ensemble": True,
    "route_pairing_mode": "compatible",
    "route_parent_weight_model": "population",
    "route_temperature_K": 303.0,
    "basin_cluster_mode": "geometry",
    "basin_energy_cluster_tol_eV": 0.0,
    "basin_geometry_rmsd_tol_A": 0.25,
    "basin_geometry_site_tol_A": None,
    "basin_local_env_enabled": True,
    "basin_local_env_cutoff_A": 3.5,
    "basin_local_env_rmsd_tol_A": 0.20,
    "basin_weight_source": "trajectory",
}

_KB_EV_PER_K = 8.617333262145e-5
_ROUTE_WEIGHT_MODEL = "empirical_conditional_route_probability"


def default_che_config() -> dict[str, object]:
    return deepcopy(_DEFAULT_CHE_CONFIG)


def _deep_update(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


class OERCHESummarizer:
    """Build site-conditioned OER free-energy changes from relaxed states."""

    def __init__(self, config: SimpleNamespace):
        self.config = config
        raw = getattr(config, "che", {}) or {}
        merged = default_che_config()
        if isinstance(raw, dict):
            merged = _deep_update(merged, raw)
        self.che_config = merged
        self._vibration_index: dict[tuple[str, str, str], dict[str, object]] | None = None
        self._reference_cache: dict[str, float] = {}

    def enabled(self) -> bool:
        return bool(self.che_config.get("enabled", False))

    def summarize(self, candidate_manifest_path: str | Path) -> dict[str, str]:
        if not self.enabled():
            return {}
        self._validate_references()
        self._vibration_rows_by_candidate()

        candidate_manifest_path = Path(candidate_manifest_path)
        if not candidate_manifest_path.exists():
            return {}

        candidate_rows = self._read_csv(candidate_manifest_path)
        site_rows = self._site_rows(candidate_rows)
        shared_clean = self._select_shared_clean_state(site_rows)
        route_rows: list[dict[str, object]] = []
        state_rows: list[dict[str, object]] = []

        for site_id, rows in site_rows.items():
            by_state = {str(row.get("state_dir", "")): row for row in rows}
            selected = self._select_site_states(site_id, by_state, shared_clean)
            selected_electronic = self._select_site_electronic_states(
                site_id,
                by_state,
                shared_clean,
                boltzmann=False,
            )
            if selected["ready"]:
                min_row = self._che_row(site_id, by_state, selected)
            else:
                min_row = self._incomplete_row(site_id, by_state, selected)
            if selected_electronic["ready"]:
                min_electronic_row = self._che_row(
                    site_id,
                    by_state,
                    selected_electronic,
                    electronic_only=True,
                )
            else:
                min_electronic_row = self._incomplete_row(
                    site_id,
                    by_state,
                    selected_electronic,
                )
            explicit_rows = self._explicit_route_rows(
                site_id,
                by_state,
                shared_clean,
            )
            if explicit_rows and min_electronic_row is not None:
                for row in explicit_rows:
                    self._fill_route_electronic_columns(
                        row,
                        "min",
                        min_electronic_row,
                    )
            route_rows.extend(
                explicit_rows
                or [
                    self._route_output_row(
                        min_row,
                        None,
                        min_electronic_row,
                        None,
                    )
                ]
            )
            state_rows.extend(
                self._state_output_rows(
                    site_id,
                    selected,
                    None,
                    selected_electronic,
                    None,
                )
            )

        self._normalize_route_weights(route_rows)
        routes_path = output_path(self.config, "oer_routes_csv")
        write_csv(routes_path, route_rows)
        states_path = output_path(self.config, "oer_states_csv")
        write_csv(states_path, state_rows)
        outputs = {
            "oer_routes_csv": str(routes_path),
            "oer_states_csv": str(states_path),
        }
        ensemble_rows = self._ensemble_summary_rows(route_rows)
        if ensemble_rows:
            ensemble_path = output_path(self.config, "oer_ensemble_csv")
            write_csv(ensemble_path, ensemble_rows)
            outputs["oer_ensemble_csv"] = str(ensemble_path)
        return outputs

    def _validate_references(self) -> None:
        self._clean_reference_mode()
        self._boltzmann_temperature()
        self._oer_reference_mode()
        missing = [
            key
            for key in ("h2_energy_eV", "h2o_energy_eV")
            if self.che_config.get(key) is None
        ]
        if (
            self._oer_reference_mode() == "explicit_o2"
            and self.che_config.get("o2_energy_eV") is None
        ):
            missing.append("o2_energy_eV")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "CHE summary requires reference energies when che.enabled=true: "
                f"missing {joined}."
            )

    def _boltzmann_enabled(self) -> bool:
        return bool(self.che_config.get("boltzmann_weight_states", False))

    def _route_ensemble_enabled(self) -> bool:
        return bool(self.che_config.get("route_ensemble", True))

    def _use_vibrational_free_energies(self) -> bool:
        return bool(self.che_config.get("use_vibrational_free_energies", True))

    def _vibration_summary_path(self) -> Path:
        value = self.che_config.get("vibration_summary_csv")
        if value not in (None, ""):
            return Path(str(value))
        return output_path(self.config, "vibration_summary_csv")

    def _vibration_rows_by_candidate(self) -> dict[tuple[str, str, str], dict[str, object]]:
        if self._vibration_index is not None:
            return self._vibration_index
        self._vibration_index = {}
        if not self._use_vibrational_free_energies():
            return self._vibration_index
        path = self._vibration_summary_path()
        if not path.exists():
            return self._vibration_index
        for row in self._read_csv(path):
            if not self._as_bool(row.get("ready")):
                continue
            energy = self._as_float(row.get("harmonic_free_energy_eV"))
            if not math.isfinite(energy):
                continue
            key = (
                str(row.get("site_id", "")),
                str(row.get("state_dir", "")),
                str(row.get("candidate_id", "")),
            )
            self._vibration_index[key] = dict(row)
        return self._vibration_index

    def _boltzmann_temperature(self) -> float:
        temperature = float(
            self.che_config.get(
                "route_temperature_K",
                self.che_config.get("boltzmann_temperature_K", 303.0),
            )
        )
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("che.route_temperature_K must be a positive number.")
        return temperature

    def _basin_energy_cluster_tol(self) -> float:
        tol = float(
            self.che_config.get(
                "basin_energy_cluster_tol_eV",
                self.che_config.get("boltzmann_energy_cluster_tol_eV", 0.0),
            )
        )
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.basin_energy_cluster_tol_eV must be a non-negative number."
            )
        return tol

    def _basin_cluster_mode(self) -> str:
        raw = str(
            self.che_config.get(
                "basin_cluster_mode",
                self.che_config.get("boltzmann_cluster_mode", "geometry"),
            )
        ).lower()
        aliases = {
            "none": "none",
            "off": "none",
            "false": "none",
            "energy": "energy",
            "energetic": "energy",
            "geometry": "geometry",
            "geometric": "geometry",
            "motif": "geometry",
            "motifs": "geometry",
            "geometry_motif": "geometry",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "che.basin_cluster_mode must be 'geometry', 'energy', or 'none'."
            )
        return mode

    def _basin_geometry_rmsd_tol(self) -> float:
        tol = float(
            self.che_config.get(
                "basin_geometry_rmsd_tol_A",
                self.che_config.get("boltzmann_geometry_rmsd_tol_A", 0.25),
            )
        )
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.basin_geometry_rmsd_tol_A must be a non-negative number."
            )
        return tol

    def _basin_geometry_site_tol(self) -> float:
        value = self.che_config.get(
            "basin_geometry_site_tol_A",
            self.che_config.get("boltzmann_geometry_site_tol_A"),
        )
        if value in (None, ""):
            value = getattr(self.config, "site_match_tol", 0.6)
        tol = float(value)
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.basin_geometry_site_tol_A must be a non-negative number."
            )
        return tol

    def _basin_local_env_enabled(self) -> bool:
        return bool(self.che_config.get("basin_local_env_enabled", True))

    def _basin_local_env_cutoff(self) -> float:
        tol = float(self.che_config.get("basin_local_env_cutoff_A", 3.5))
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.basin_local_env_cutoff_A must be a non-negative number."
            )
        return tol

    def _basin_local_env_rmsd_tol(self) -> float:
        tol = float(self.che_config.get("basin_local_env_rmsd_tol_A", 0.20))
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.basin_local_env_rmsd_tol_A must be a non-negative number."
            )
        return tol

    def _basin_weight_source(self) -> str:
        raw = str(self.che_config.get("basin_weight_source", "trajectory")).strip().lower()
        aliases = {
            "trajectory": "trajectory",
            "traj": "trajectory",
            "full_trajectory": "trajectory",
            "samples": "selected",
            "selected": "selected",
            "selected_candidates": "selected",
            "candidate": "selected",
            "candidates": "selected",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "che.basin_weight_source must be 'trajectory' or 'selected'."
            )
        return mode

    def _boltzmann_energy_cluster_tol(self) -> float:
        return self._basin_energy_cluster_tol()

    def _boltzmann_cluster_mode(self) -> str:
        return self._basin_cluster_mode()

    def _boltzmann_geometry_rmsd_tol(self) -> float:
        return self._basin_geometry_rmsd_tol()

    def _boltzmann_geometry_site_tol(self) -> float:
        return self._basin_geometry_site_tol()

    def _route_pairing_mode(self) -> str:
        raw = str(self.che_config.get("route_pairing_mode", "compatible")).lower()
        aliases = {
            "compatible": "compatible",
            "linked": "compatible",
            "parent_o": "compatible",
            "cartesian": "cartesian",
            "all": "cartesian",
            "product": "cartesian",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "che.route_pairing_mode must be 'compatible' or 'cartesian'."
            )
        return mode

    def _clean_reference_mode(self) -> str:
        raw = str(self.che_config.get("clean_reference_mode", "per_site")).strip().lower()
        aliases = {
            "site": "per_site",
            "per-site": "per_site",
            "per_site": "per_site",
            "local": "per_site",
            "shared": "lowest_energy",
            "shared_lowest": "lowest_energy",
            "shared_lowest_energy": "lowest_energy",
            "lowest": "lowest_energy",
            "lowest_energy": "lowest_energy",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "che.clean_reference_mode must be 'per_site' or 'lowest_energy'."
            )
        return mode

    def _oer_reference_mode(self) -> str:
        raw = str(self.che_config.get("oer_reference_mode", "auto")).strip().lower()
        aliases = {
            "auto": (
                "explicit_o2"
                if self.che_config.get("o2_energy_eV") is not None
                else "closure"
            ),
            "closure": "closure",
            "total": "closure",
            "total_oer": "closure",
            "4.92": "closure",
            "explicit": "explicit_o2",
            "explicit_o2": "explicit_o2",
            "o2": "explicit_o2",
        }
        mode = aliases.get(raw)
        if mode is None:
            raise ValueError(
                "che.oer_reference_mode must be 'auto', 'closure', or 'explicit_o2'."
            )
        return mode

    def _select_shared_clean_state(
        self,
        site_rows: dict[str, list[dict[str, object]]],
    ) -> dict[str, object] | None:
        if self._clean_reference_mode() != "lowest_energy":
            return None

        clean_state = str(self.che_config.get("clean_state", "00_clean"))
        candidates: list[dict[str, object]] = []
        for site_id, rows in site_rows.items():
            by_state = {str(row.get("state_dir", "")): row for row in rows}
            manifest_row = by_state.get(clean_state)
            if manifest_row is None:
                continue
            best = self._best_energy_row(manifest_row)
            if best is None:
                continue
            candidates.append(
                self._annotate_selected_state(
                    best,
                    manifest_row,
                    source_site_id=site_id,
                    reference_mode="lowest_energy",
                )
            )
        if not candidates:
            raise ValueError(
                "che.clean_reference_mode=lowest_energy requires at least one "
                f"relaxed {clean_state} energies.csv entry."
            )
        return min(candidates, key=lambda row: self._as_float(row.get("energy_eV")))

    def _site_rows(
        self,
        candidate_rows: Sequence[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in candidate_rows:
            grouped.setdefault(str(row.get("site_id", "")), []).append(dict(row))
        return grouped

    def _select_site_states(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        shared_clean: dict[str, object] | None = None,
    ) -> dict[str, object]:
        state_map = {
            "clean": str(self.che_config.get("clean_state", "00_clean")),
            "oh": str(self.che_config.get("oh_state", "01_OH")),
            "o": str(self.che_config.get("o_state", "02_O")),
            "ooh": str(self.che_config.get("ooh_state", "03_OOH")),
        }
        selected: dict[str, dict[str, object]] = {}
        missing: list[str] = []

        for label, state_dir in state_map.items():
            if label == "clean" and shared_clean is not None:
                selected[label] = dict(shared_clean)
                continue
            manifest_row = by_state.get(state_dir)
            if manifest_row is None:
                missing.append(state_dir)
                continue
            best = self._best_energy_row(manifest_row)
            if best is None:
                missing.append(state_dir)
                continue
            best = self._annotate_selected_state(
                best,
                manifest_row,
                source_site_id=site_id,
                reference_mode="per_site",
            )
            selected[label] = best

        return {
            "ready": len(missing) == 0,
            "missing_states": missing,
            "states": selected,
        }

    def _select_site_boltzmann_states(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        shared_clean: dict[str, object] | None = None,
    ) -> dict[str, object]:
        state_map = {
            "clean": str(self.che_config.get("clean_state", "00_clean")),
            "oh": str(self.che_config.get("oh_state", "01_OH")),
            "o": str(self.che_config.get("o_state", "02_O")),
            "ooh": str(self.che_config.get("ooh_state", "03_OOH")),
        }
        selected: dict[str, dict[str, object]] = {}
        counts: dict[str, int] = {}
        missing: list[str] = []

        for label, state_dir in state_map.items():
            if label == "clean" and shared_clean is not None:
                selected[label] = dict(shared_clean)
                counts[label] = 1
                continue
            manifest_row = by_state.get(state_dir)
            if manifest_row is None:
                missing.append(state_dir)
                continue
            if label == "clean":
                state = self._best_energy_row(manifest_row)
            else:
                state = self._boltzmann_energy_row(manifest_row)
            if state is None:
                missing.append(state_dir)
                continue
            selected[label] = self._annotate_selected_state(
                state,
                manifest_row,
                source_site_id=site_id,
                reference_mode="boltzmann" if label != "clean" else "per_site",
            )
            counts[label] = int(state.get("n_boltzmann_states", 1))

        return {
            "ready": len(missing) == 0,
            "missing_states": missing,
            "states": selected,
            "state_counts": counts,
        }

    def _select_site_electronic_states(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        shared_clean: dict[str, object] | None = None,
        *,
        boltzmann: bool,
    ) -> dict[str, object]:
        state_map = {
            "clean": str(self.che_config.get("clean_state", "00_clean")),
            "oh": str(self.che_config.get("oh_state", "01_OH")),
            "o": str(self.che_config.get("o_state", "02_O")),
            "ooh": str(self.che_config.get("ooh_state", "03_OOH")),
        }
        selected: dict[str, dict[str, object]] = {}
        counts: dict[str, int] = {}
        missing: list[str] = []

        for label, state_dir in state_map.items():
            if label == "clean" and shared_clean is not None:
                state = dict(shared_clean)
                energy = self._as_float(state.get("electronic_energy_eV"))
                if not math.isfinite(energy):
                    missing.append(state_dir)
                    continue
                state["energy_eV"] = energy
                state["energy_source"] = "electronic"
                state["_reference_mode"] = "lowest_energy"
                selected[label] = state
                counts[label] = 1
                continue

            manifest_row = by_state.get(state_dir)
            if manifest_row is None:
                missing.append(state_dir)
                continue
            if boltzmann and label != "clean":
                state = self._boltzmann_electronic_energy_row(manifest_row)
            else:
                state = self._best_electronic_energy_row(manifest_row)
            if state is None:
                missing.append(state_dir)
                continue
            selected[label] = self._annotate_selected_state(
                state,
                manifest_row,
                source_site_id=site_id,
                reference_mode="boltzmann_electronic" if boltzmann and label != "clean" else "electronic",
            )
            counts[label] = int(state.get("n_boltzmann_states", 1))

        return {
            "ready": len(missing) == 0,
            "missing_states": missing,
            "states": selected,
            "state_counts": counts,
        }

    def _explicit_route_rows(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        shared_clean: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        if not self._route_ensemble_enabled():
            return []
        state_map = {
            "clean": str(self.che_config.get("clean_state", "00_clean")),
            "oh": str(self.che_config.get("oh_state", "01_OH")),
            "o": str(self.che_config.get("o_state", "02_O")),
            "ooh": str(self.che_config.get("ooh_state", "03_OOH")),
        }
        clean = shared_clean
        if clean is None:
            clean_manifest = by_state.get(state_map["clean"])
            if clean_manifest is None:
                return []
            clean = self._best_energy_row(clean_manifest)
            if clean is None:
                return []
            clean = self._annotate_selected_state(
                clean,
                clean_manifest,
                source_site_id=site_id,
                reference_mode="per_site",
            )

        oh_manifest = by_state.get(state_map["oh"])
        if oh_manifest is None:
            return []
        oh = self._best_energy_row(oh_manifest)
        if oh is None:
            return []
        oh = self._annotate_selected_state(
            oh,
            oh_manifest,
            source_site_id=site_id,
            reference_mode="route_parent",
        )

        o_manifest = by_state.get(state_map["o"])
        ooh_manifest = by_state.get(state_map["ooh"])
        if o_manifest is None or ooh_manifest is None:
            return []
        o_basins = self._basin_energy_rows(o_manifest)
        ooh_basins = self._basin_energy_rows(ooh_manifest)
        if not o_basins or not ooh_basins:
            return []

        site_manifest = next(iter(by_state.values()), {})
        parent_weight = self._as_float(self._site_population(site_manifest))
        if not math.isfinite(parent_weight) or parent_weight <= 0.0:
            parent_weight = 1.0
        o_total = sum(max(0, int(row.get("basin_count", 1))) for row in o_basins)
        if o_total <= 0:
            o_total = len(o_basins)

        rows: list[dict[str, object]] = []
        for o_index, o_state in enumerate(o_basins, start=1):
            o_count = max(0, int(o_state.get("basin_count", 1))) or 1
            o_probability = o_count / o_total
            compatible_ooh = self._compatible_ooh_basins(o_state, ooh_basins)
            if not compatible_ooh:
                continue
            ooh_total = sum(
                max(0, int(row.get("basin_count", 1))) for row in compatible_ooh
            )
            if ooh_total <= 0:
                ooh_total = len(compatible_ooh)
            for ooh_index, ooh_state in enumerate(compatible_ooh, start=1):
                ooh_count = max(0, int(ooh_state.get("basin_count", 1))) or 1
                ooh_probability = ooh_count / ooh_total
                selected = {
                    "ready": True,
                    "missing_states": [],
                    "states": {
                        "clean": clean,
                        "oh": oh,
                        "o": o_state,
                        "ooh": ooh_state,
                    },
                }
                base = self._che_row(site_id, by_state, selected)
                rows.append(
                    self._explicit_route_output_row(
                        base,
                        oh,
                        o_state,
                        ooh_state,
                        o_index=o_index,
                        ooh_index=ooh_index,
                        parent_weight=parent_weight,
                        o_probability=o_probability,
                        ooh_probability=ooh_probability,
                        n_o_basins=len(o_basins),
                        n_ooh_basins=len(compatible_ooh),
                    )
                )
        return rows

    def _basin_energy_rows(
        self,
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]]:
        _, finite, _ = self._finite_energy_rows(manifest_row)
        if not finite:
            return []
        return self._cluster_basin_rows(finite, manifest_row)

    def _cluster_basin_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]]:
        mode = self._basin_cluster_mode()
        if mode == "geometry":
            clustered = self._cluster_geometry_basin_rows(rows, manifest_row)
            if clustered is not None:
                return clustered
        if mode == "energy":
            return self._cluster_energy_basin_rows(rows, manifest_row)
        return self._finalize_basin_rows(rows, manifest_row)

    def _cluster_energy_basin_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]]:
        tol = self._basin_energy_cluster_tol()
        sorted_rows = sorted(rows, key=lambda row: self._as_float(row.get("energy_eV")))
        clusters: list[dict[str, object]] = []
        cluster_energies: list[float] = []
        for row in sorted_rows:
            energy = self._as_float(row.get("energy_eV"))
            match = None
            if tol > 0.0:
                for index, reference in enumerate(cluster_energies):
                    if abs(energy - reference) <= tol:
                        match = index
                        break
            if match is None:
                clusters.append(dict(row))
                cluster_energies.append(energy)
            else:
                self._add_to_basin_cluster(clusters[match], row)
        return self._finalize_basin_rows(clusters, manifest_row)

    def _cluster_geometry_basin_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]] | None:
        state_path = Path(str(manifest_row.get("candidates_csv", ""))).parent
        relaxed_traj = state_path / "relaxed.traj"
        candidates_csv = state_path / "candidates.csv"
        if not relaxed_traj.exists():
            return None
        try:
            structures = self._read_atoms_list(relaxed_traj)
        except Exception:
            return None
        candidate_rows = self._read_csv(candidates_csv) if candidates_csv.exists() else []
        if not structures:
            return None

        raw_structures: list | None = None
        candidates_traj = state_path / "candidates.traj"
        if candidates_traj.exists():
            try:
                raw_structures = self._read_atoms_list(candidates_traj)
            except Exception:
                raw_structures = None

        clusters: list[dict[str, object]] = []
        descriptors: list[dict[str, object]] = []
        for row in sorted(rows, key=lambda item: self._as_float(item.get("energy_eV"))):
            candidate_index = self._candidate_index(row)
            if not (0 <= candidate_index < len(structures)):
                return None
            candidate_row = (
                candidate_rows[candidate_index]
                if candidate_index < len(candidate_rows)
                else {}
            )
            descriptor = self._geometry_descriptor(
                structures[candidate_index],
                row,
                candidate_row,
            )
            if descriptor is None:
                return None
            match = None
            for index, existing in enumerate(descriptors):
                if self._same_geometry_cluster(descriptor, existing):
                    match = index
                    break
            if match is None:
                output = dict(row)
                output["geometry_cluster_key"] = descriptor["site_key"]
                clusters.append(output)
                descriptors.append(descriptor)
            else:
                self._add_to_basin_cluster(clusters[match], row)
        self._assign_basin_counts_from_trajectories(
            clusters,
            descriptors,
            candidate_rows,
            structures,
            raw_structures,
        )
        return self._finalize_basin_rows(clusters, manifest_row)

    def _finalize_basin_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]]:
        state_dir = str(manifest_row.get("state_dir", "state"))
        total = 0
        output_rows: list[dict[str, object]] = []
        for index, row in enumerate(rows, start=1):
            output = dict(row)
            count = max(1, int(output.get("basin_count", 1)))
            output["basin_count"] = count
            output["basin_id"] = output.get("basin_id") or f"{state_dir}:basin{index:03d}"
            output["basin_candidate_ids"] = output.get(
                "basin_candidate_ids",
                str(output.get("candidate_id", "")),
            )
            output["basin_representative_candidate_id"] = output.get("candidate_id", "")
            output = self._annotate_selected_state(
                output,
                manifest_row,
                source_site_id=str(manifest_row.get("site_id", "")),
                reference_mode="basin_representative",
            )
            total += count
            output_rows.append(output)
        if total <= 0:
            total = len(output_rows)
        for row in output_rows:
            row["basin_probability"] = int(row.get("basin_count", 1)) / total
        return output_rows

    def _assign_basin_counts_from_trajectories(
        self,
        clusters: list[dict[str, object]],
        descriptors: list[dict[str, object]],
        candidate_rows: Sequence[dict[str, object]],
        structures: Sequence,
        raw_structures: Sequence | None,
    ) -> None:
        if self._basin_weight_source() != "trajectory":
            return
        if not clusters or len(clusters) != len(descriptors):
            return

        representative_descriptors = [
            self._assignment_descriptor(
                self._representative_structure(
                    cluster,
                    structures,
                    raw_structures,
                )
            )
            for cluster in clusters
        ]
        if any(item is None for item in representative_descriptors):
            return

        trajectory_sources = self._basin_assignment_frame_sources(candidate_rows)
        if not trajectory_sources:
            return

        counts = [0 for _ in clusters]
        assigned = 0
        total = 0
        for frame in self._iter_assignment_frames(trajectory_sources):
            total += 1
            descriptor = self._assignment_descriptor(frame)
            if descriptor is None:
                continue
            distances = [
                self._assignment_descriptor_distance(descriptor, representative)
                for representative in representative_descriptors
            ]
            if not distances or not any(math.isfinite(value) for value in distances):
                continue
            index = min(range(len(distances)), key=lambda item: distances[item])
            counts[index] += 1
            assigned += 1

        if total <= 0 or assigned <= 0:
            return

        for cluster, count in zip(clusters, counts):
            cluster["basin_selected_count"] = int(cluster.get("basin_count", 1))
            cluster["basin_count"] = int(count)
            cluster["basin_trajectory_count"] = int(count)
            cluster["basin_trajectory_total_count"] = int(total)
            cluster["basin_trajectory_assigned_count"] = int(assigned)
            cluster["basin_trajectory_unassigned_count"] = int(total - assigned)
            cluster["basin_weight_source"] = "trajectory"

    def _representative_structure(
        self,
        cluster: dict[str, object],
        structures: Sequence,
        raw_structures: Sequence | None,
    ):
        candidate_index = self._candidate_index(cluster)
        if raw_structures is not None and 0 <= candidate_index < len(raw_structures):
            return raw_structures[candidate_index]
        if 0 <= candidate_index < len(structures):
            return structures[candidate_index]
        return None

    def _assignment_descriptor(self, atoms) -> dict[str, object] | None:
        if atoms is None:
            return None
        ads_indices = self._adsorbate_indices(atoms, {})
        if not ads_indices:
            return None
        anchor_idx = int(ads_indices[0])
        rel = np.asarray(
            atoms.get_distances(anchor_idx, ads_indices, mic=True, vector=True),
            dtype=float,
        )
        return {
            "symbols": tuple(atoms[int(idx)].symbol for idx in ads_indices),
            "anchor_xy": np.asarray(atoms.positions[anchor_idx, :2], dtype=float),
            "relative_positions": rel,
            "local_environment": self._local_environment_descriptor(
                atoms,
                ads_indices,
                anchor_idx,
            ),
            "cell": np.asarray(atoms.get_cell(), dtype=float),
            "pbc": np.asarray(atoms.get_pbc(), dtype=bool),
        }

    def _assignment_descriptor_distance(
        self,
        left: dict[str, object],
        right: dict[str, object],
    ) -> float:
        if left.get("symbols") != right.get("symbols"):
            return float("inf")
        rel_left = np.asarray(left.get("relative_positions", ()), dtype=float)
        rel_right = np.asarray(right.get("relative_positions", ()), dtype=float)
        if rel_left.shape != rel_right.shape:
            return float("inf")
        anchor_distance = self._assignment_anchor_distance(left, right)
        adsorbate_rmsd = float(
            np.sqrt(np.mean(np.sum((rel_left - rel_right) ** 2, axis=1)))
        )
        distance = anchor_distance + adsorbate_rmsd
        env_distance = self._local_environment_distance(
            left.get("local_environment"),
            right.get("local_environment"),
        )
        if math.isfinite(env_distance):
            distance += env_distance
        return float(distance)

    @staticmethod
    def _assignment_anchor_distance(
        left: dict[str, object],
        right: dict[str, object],
    ) -> float:
        xy_left = np.asarray(left.get("anchor_xy", (np.nan, np.nan)), dtype=float)
        xy_right = np.asarray(right.get("anchor_xy", (np.nan, np.nan)), dtype=float)
        if xy_left.shape[0] < 2 or xy_right.shape[0] < 2:
            return float("inf")
        point_left = np.array([xy_left[0], xy_left[1], 0.0], dtype=float)
        point_right = np.array([xy_right[0], xy_right[1], 0.0], dtype=float)
        pbc = np.asarray(left.get("pbc", (False, False, False)), dtype=bool)
        if any(pbc[:2]):
            delta = get_distances(
                point_left.reshape(1, 3),
                point_right.reshape(1, 3),
                cell=np.asarray(left.get("cell"), dtype=float),
                pbc=pbc,
            )[0][0, 0]
            return float(np.linalg.norm(delta[:2]))
        return float(np.linalg.norm(point_left[:2] - point_right[:2]))

    def _raw_candidate_descriptor(
        self,
        cluster: dict[str, object],
        candidate_rows: Sequence[dict[str, object]],
        raw_structures: Sequence | None,
    ) -> dict[str, object] | None:
        if raw_structures is None:
            return None
        candidate_index = self._candidate_index(cluster)
        if not (0 <= candidate_index < len(raw_structures)):
            return None
        candidate_row = (
            candidate_rows[candidate_index]
            if candidate_index < len(candidate_rows)
            else {}
        )
        return self._geometry_descriptor(
            raw_structures[candidate_index],
            cluster,
            candidate_row,
        )

    def _basin_assignment_frame_sources(
        self,
        candidate_rows: Sequence[dict[str, object]],
    ) -> list[Path]:
        sources: list[Path] = []
        seen: set[Path] = set()
        for row in candidate_rows:
            path = self._candidate_assignment_trajectory(row)
            if path is None or path in seen or not path.exists():
                continue
            sources.append(path)
            seen.add(path)
        return sources

    def _iter_assignment_frames(self, sources: Sequence[Path]):
        for path in sources:
            try:
                yield from iread(str(path), index=":")
            except Exception:
                continue

    def _candidate_assignment_trajectory(
        self,
        candidate_row: dict[str, object],
    ) -> Path | None:
        pt_dir = str(candidate_row.get("local_cmc_pt_dir", "")).strip()
        if pt_dir:
            temperature = self._as_float(
                candidate_row.get("local_cmc_pt_temperature_K")
            )
            if math.isfinite(temperature):
                return Path(pt_dir) / f"replica_{self._temperature_label(temperature)}.traj"

        seed_index = candidate_row.get("local_cmc_seed_index")
        if seed_index in (None, ""):
            return None
        try:
            seed = int(float(seed_index))
        except (TypeError, ValueError):
            return None
        csv_path = candidate_row.get("local_cmc_dir")
        if csv_path in (None, ""):
            return None
        return Path(str(csv_path)) / f"seed{seed:03d}_samples.traj"

    def _frame_anchor_index(self, atoms) -> int:
        ads_indices = self._adsorbate_indices(atoms, {})
        if ads_indices:
            return int(ads_indices[0])
        return -1

    def _frame_representative_distance(self, frame, representative) -> float:
        frame_ads = self._adsorbate_indices(frame, {})
        representative_ads = self._adsorbate_indices(representative, {})
        if not frame_ads or not representative_ads:
            return float("inf")
        if len(frame_ads) != len(representative_ads):
            return float("inf")
        frame_symbols = tuple(frame[int(idx)].symbol for idx in frame_ads)
        representative_symbols = tuple(
            representative[int(idx)].symbol for idx in representative_ads
        )
        if frame_symbols != representative_symbols:
            return float("inf")

        frame_anchor = int(frame_ads[0])
        representative_anchor = int(representative_ads[0])
        anchor_distance = self._xy_distance(
            frame,
            np.asarray(frame.positions[frame_anchor, :2], dtype=float),
            np.asarray(representative.positions[representative_anchor, :2], dtype=float),
        )
        frame_rel = np.asarray(
            frame.get_distances(frame_anchor, frame_ads, mic=True, vector=True),
            dtype=float,
        )
        representative_rel = np.asarray(
            representative.get_distances(
                representative_anchor,
                representative_ads,
                mic=True,
                vector=True,
            ),
            dtype=float,
        )
        if frame_rel.shape != representative_rel.shape:
            return float("inf")
        adsorbate_rmsd = float(
            np.sqrt(np.mean(np.sum((frame_rel - representative_rel) ** 2, axis=1)))
        )

        distance = float(anchor_distance + adsorbate_rmsd)
        env_distance = self._local_environment_distance(
            self._local_environment_descriptor(frame, frame_ads, frame_anchor),
            self._local_environment_descriptor(
                representative,
                representative_ads,
                representative_anchor,
            ),
        )
        if math.isfinite(env_distance):
            distance += env_distance
        return distance

    @staticmethod
    def _temperature_label(temperature: float) -> str:
        text = f"{float(temperature):.6g}".replace(".", "p")
        return f"{text}K"

    @staticmethod
    def _add_to_basin_cluster(cluster: dict[str, object], row: dict[str, object]) -> None:
        cluster["basin_count"] = int(cluster.get("basin_count", 1)) + 1
        existing = str(cluster.get("basin_candidate_ids", cluster.get("candidate_id", "")))
        candidate_id = str(row.get("candidate_id", ""))
        values = [value for value in existing.split() if value]
        if candidate_id and candidate_id not in values:
            values.append(candidate_id)
        cluster["basin_candidate_ids"] = " ".join(values)

    def _compatible_ooh_basins(
        self,
        o_state: dict[str, object],
        ooh_basins: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        if self._route_pairing_mode() == "cartesian":
            return [dict(row) for row in ooh_basins]
        linked = [
            row
            for row in ooh_basins
            if self._parent_candidate_id(row)
        ]
        if not linked:
            return [dict(row) for row in ooh_basins]
        return [
            dict(row)
            for row in linked
            if self._ooh_matches_o_basin(o_state, row)
        ]

    @staticmethod
    def _ooh_matches_o_basin(
        o_state: dict[str, object],
        ooh_state: dict[str, object],
    ) -> bool:
        parent_id = OERCHESummarizer._parent_candidate_id(ooh_state)
        if not parent_id:
            return True
        ids = {
            str(o_state.get("candidate_id", "")).strip(),
            str(o_state.get("basin_representative_candidate_id", "")).strip(),
        }
        ids.update(
            value.strip()
            for value in str(o_state.get("basin_candidate_ids", "")).split()
            if value.strip()
        )
        for candidate_id in ids:
            if not candidate_id:
                continue
            if parent_id == candidate_id:
                return True
            if candidate_id.startswith(parent_id + "_"):
                return True
            if parent_id.startswith(candidate_id + "_"):
                return True
        return False

    @staticmethod
    def _parent_candidate_id(row: dict[str, object]) -> str:
        for key in (
            "parent_state_candidate_id",
            "parent_candidate_id",
            "parent_o_candidate_id",
        ):
            value = str(row.get(key, "")).strip()
            if value:
                return value
        return ""

    def _annotate_selected_state(
        self,
        best: dict[str, object],
        manifest_row: dict[str, object],
        *,
        source_site_id: str,
        reference_mode: str,
    ) -> dict[str, object]:
        output = dict(best)
        output["_source_site_id"] = source_site_id
        output["_source_site_dir"] = manifest_row.get("site_dir", "")
        output["_source_state_dir"] = manifest_row.get("state_dir", "")
        output["_source_species"] = manifest_row.get("species", "")
        output["_reference_mode"] = reference_mode
        return output

    def _best_energy_row(
        self,
        manifest_row: dict[str, object],
    ) -> dict[str, object] | None:
        energies_csv, finite, _ = self._finite_energy_rows(manifest_row)
        if not finite:
            return None
        best = min(finite, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energies_csv"] = str(energies_csv)
        output["used_unconverged_fallback"] = bool(
            self.che_config.get("use_converged_only", True)
            and not self._as_bool(output.get("converged"))
        )
        return output

    def _boltzmann_energy_row(
        self,
        manifest_row: dict[str, object],
    ) -> dict[str, object] | None:
        energies_csv, finite, _ = self._finite_energy_rows(manifest_row)
        if not finite:
            return None
        clustered = self._cluster_boltzmann_rows(finite, manifest_row)
        energies = [self._as_float(row.get("energy_eV")) for row in clustered]
        free_energy = self._logsumexp_free_energy(energies)
        best = min(clustered, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energy_eV"] = free_energy
        output["energies_csv"] = str(energies_csv)
        output["n_boltzmann_states"] = len(clustered)
        output["n_raw_boltzmann_states"] = len(finite)
        output["boltzmann_cluster_mode"] = self._boltzmann_cluster_mode()
        output["min_energy_eV"] = min(energies)
        output["used_unconverged_fallback"] = bool(
            self.che_config.get("use_converged_only", True)
            and not self._as_bool(output.get("converged"))
        )
        return output

    def _best_electronic_energy_row(
        self,
        manifest_row: dict[str, object],
    ) -> dict[str, object] | None:
        energies_csv, finite, _ = self._finite_electronic_energy_rows(manifest_row)
        if not finite:
            return None
        best = min(finite, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energies_csv"] = str(energies_csv)
        output["used_unconverged_fallback"] = bool(
            self.che_config.get("use_converged_only", True)
            and not self._as_bool(output.get("converged"))
        )
        return output

    def _boltzmann_electronic_energy_row(
        self,
        manifest_row: dict[str, object],
    ) -> dict[str, object] | None:
        energies_csv, finite, _ = self._finite_electronic_energy_rows(manifest_row)
        if not finite:
            return None
        clustered = self._cluster_boltzmann_rows(finite, manifest_row)
        energies = [self._as_float(row.get("energy_eV")) for row in clustered]
        free_energy = self._logsumexp_free_energy(energies)
        best = min(clustered, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energy_eV"] = free_energy
        output["energies_csv"] = str(energies_csv)
        output["n_boltzmann_states"] = len(clustered)
        output["n_raw_boltzmann_states"] = len(finite)
        output["boltzmann_cluster_mode"] = self._boltzmann_cluster_mode()
        output["min_energy_eV"] = min(energies)
        output["used_unconverged_fallback"] = bool(
            self.che_config.get("use_converged_only", True)
            and not self._as_bool(output.get("converged"))
        )
        return output

    def _finite_electronic_energy_rows(
        self,
        manifest_row: dict[str, object],
    ) -> tuple[Path, list[dict[str, object]], bool]:
        return self._finite_energy_rows_with_builder(
            manifest_row,
            self._effective_electronic_energy_row,
        )

    def _finite_energy_rows(
        self,
        manifest_row: dict[str, object],
    ) -> tuple[Path, list[dict[str, object]], bool]:
        return self._finite_energy_rows_with_builder(
            manifest_row,
            self._effective_energy_row,
        )

    def _finite_energy_rows_with_builder(
        self,
        manifest_row: dict[str, object],
        builder,
    ) -> tuple[Path, list[dict[str, object]], bool]:
        energies_csv = Path(str(manifest_row.get("candidates_csv", ""))).parent / "energies.csv"
        if not energies_csv.exists():
            return energies_csv, [], False
        rows = self._read_csv(energies_csv)
        if not rows:
            return energies_csv, [], False

        use_converged = bool(self.che_config.get("use_converged_only", True))
        allow_fallback = bool(self.che_config.get("allow_unconverged_fallback", False))
        rows = [row for row in rows if self._row_bool_or_true(row.get("adsorbate_intact"))]
        if not rows:
            return energies_csv, [], False
        pool = rows
        used_fallback = False
        if use_converged:
            converged = [row for row in rows if self._as_bool(row.get("converged"))]
            if converged:
                pool = converged
            elif not allow_fallback:
                return energies_csv, [], False
            else:
                used_fallback = True
        enriched = [builder(manifest_row, row) for row in pool]
        finite = [row for row in enriched if math.isfinite(self._as_float(row.get("energy_eV")))]
        return energies_csv, finite, used_fallback

    def _effective_electronic_energy_row(
        self,
        manifest_row: dict[str, object],
        row: dict[str, object],
    ) -> dict[str, object]:
        output = dict(row)
        electronic_energy = self._as_float(output.get("energy_eV"))
        output["energy_eV"] = electronic_energy
        output["electronic_energy_eV"] = electronic_energy
        output["energy_source"] = "electronic"
        output["harmonic_correction_eV"] = ""
        output["vibration_summary_csv"] = ""
        if not self._use_vibrational_free_energies():
            return output
        if not self._vibration_summary_path().exists():
            return output

        key = (
            str(manifest_row.get("site_id", "")),
            str(manifest_row.get("state_dir", "")),
            str(output.get("candidate_id", "")),
        )
        vibration_row = self._vibration_rows_by_candidate().get(key)
        if vibration_row is None:
            output["energy_eV"] = math.nan
            return output
        output["vibration_summary_csv"] = str(self._vibration_summary_path())
        return output

    def _effective_energy_row(
        self,
        manifest_row: dict[str, object],
        row: dict[str, object],
    ) -> dict[str, object]:
        output = dict(row)
        electronic_energy = self._as_float(output.get("energy_eV"))
        output["electronic_energy_eV"] = electronic_energy
        output["energy_source"] = "electronic"
        output["harmonic_correction_eV"] = ""
        output["vibration_summary_csv"] = ""
        if not self._use_vibrational_free_energies():
            return output
        if not self._vibration_summary_path().exists():
            return output

        key = (
            str(manifest_row.get("site_id", "")),
            str(manifest_row.get("state_dir", "")),
            str(output.get("candidate_id", "")),
        )
        vibration_row = self._vibration_rows_by_candidate().get(key)
        if vibration_row is None:
            output["energy_eV"] = math.nan
            return output

        harmonic_energy = self._as_float(vibration_row.get("harmonic_free_energy_eV"))
        if not math.isfinite(harmonic_energy):
            output["energy_eV"] = math.nan
            return output
        output["energy_eV"] = harmonic_energy
        output["energy_source"] = "harmonic"
        output["harmonic_correction_eV"] = vibration_row.get("harmonic_correction_eV", "")
        output["vibration_summary_csv"] = str(self._vibration_summary_path())
        return output

    def _cluster_boltzmann_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]]:
        mode = self._boltzmann_cluster_mode()
        if mode == "none":
            return [dict(row) for row in rows]
        if mode == "geometry":
            clustered = self._cluster_geometry_rows(rows, manifest_row)
            if clustered is not None:
                return clustered
        return self._cluster_energy_rows(rows)

    def _cluster_energy_rows(
        self,
        rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        tol = self._boltzmann_energy_cluster_tol()
        if tol == 0.0:
            return [dict(row) for row in rows]
        sorted_rows = sorted(rows, key=lambda row: self._as_float(row.get("energy_eV")))
        clusters: list[dict[str, object]] = []
        last_energy: float | None = None
        for row in sorted_rows:
            energy = self._as_float(row.get("energy_eV"))
            if last_energy is None or abs(energy - last_energy) > tol:
                clusters.append(dict(row))
                last_energy = energy
        return clusters

    def _cluster_geometry_rows(
        self,
        rows: Sequence[dict[str, object]],
        manifest_row: dict[str, object],
    ) -> list[dict[str, object]] | None:
        state_path = Path(str(manifest_row.get("candidates_csv", ""))).parent
        relaxed_traj = state_path / "relaxed.traj"
        candidates_csv = state_path / "candidates.csv"
        if not relaxed_traj.exists():
            return None
        try:
            structures = self._read_atoms_list(relaxed_traj)
        except Exception:
            return None
        candidate_rows = self._read_csv(candidates_csv) if candidates_csv.exists() else []
        if not structures:
            return None

        clusters: list[dict[str, object]] = []
        descriptors: list[dict[str, object]] = []
        for row in sorted(rows, key=lambda item: self._as_float(item.get("energy_eV"))):
            candidate_index = self._candidate_index(row)
            if not (0 <= candidate_index < len(structures)):
                return None
            candidate_row = (
                candidate_rows[candidate_index]
                if candidate_index < len(candidate_rows)
                else {}
            )
            descriptor = self._geometry_descriptor(
                structures[candidate_index],
                row,
                candidate_row,
            )
            if descriptor is None:
                return None

            matched = False
            for existing in descriptors:
                if self._same_geometry_cluster(descriptor, existing):
                    matched = True
                    break
            if not matched:
                output = dict(row)
                output["geometry_cluster_key"] = descriptor["site_key"]
                clusters.append(output)
                descriptors.append(descriptor)
        return clusters

    def _geometry_descriptor(
        self,
        atoms,
        energy_row: dict[str, object],
        candidate_row: dict[str, object],
    ) -> dict[str, object] | None:
        ads_indices = self._adsorbate_indices(atoms, candidate_row)
        if not ads_indices:
            return None
        anchor_idx = int(self._as_float(energy_row.get("anchor_index")))
        if not (0 <= anchor_idx < len(atoms)) or anchor_idx not in ads_indices:
            anchor_idx = int(ads_indices[0])
        site_key = self._anchor_site_key(atoms, ads_indices, anchor_idx)
        rel = np.asarray(
            atoms.get_distances(anchor_idx, ads_indices, mic=True, vector=True),
            dtype=float,
        )
        symbols = tuple(atoms[int(idx)].symbol for idx in ads_indices)
        return {
            "site_key": site_key,
            "symbols": symbols,
            "relative_positions": rel,
            "local_environment": self._local_environment_descriptor(
                atoms,
                ads_indices,
                anchor_idx,
            ),
        }

    def _local_environment_descriptor(
        self,
        atoms,
        ads_indices: Sequence[int],
        anchor_idx: int,
    ) -> dict[str, object] | None:
        if not self._basin_local_env_enabled():
            return None

        cutoff = self._basin_local_env_cutoff()
        if cutoff <= 0.0:
            return None

        ads_set = {int(idx) for idx in ads_indices}
        elements = set(getattr(self.config, "substrate_elements", ()) or ())
        elements.update(getattr(self.config, "functional_elements", ()) or ())
        elements.update(getattr(self.config, "site_elements", ()) or ())
        if not elements:
            elements = {atom.symbol for atom in atoms}

        indices: list[int] = []
        symbols: list[str] = []
        for idx, atom in enumerate(atoms):
            if idx in ads_set or idx == anchor_idx or atom.symbol not in elements:
                continue
            indices.append(int(idx))
            symbols.append(str(atom.symbol))
        if not indices:
            return {
                "symbols": (),
                "relative_positions": np.empty((0, 3), dtype=float),
            }

        vectors = np.asarray(
            atoms.get_distances(anchor_idx, indices, mic=True, vector=True),
            dtype=float,
        )
        distances = np.linalg.norm(vectors, axis=1)
        entries: list[tuple[str, float, tuple[float, float, float]]] = []
        for symbol, distance, vector in zip(symbols, distances, vectors):
            distance = float(distance)
            if distance <= cutoff:
                entries.append(
                    (
                        symbol,
                        distance,
                        (float(vector[0]), float(vector[1]), float(vector[2])),
                    )
                )

        entries.sort(key=lambda item: (item[0], round(item[1], 8), item[2]))
        return {
            "symbols": tuple(entry[0] for entry in entries),
            "relative_positions": np.asarray(
                [entry[2] for entry in entries],
                dtype=float,
            ),
        }

    def _anchor_site_key(
        self,
        atoms,
        ads_indices: Sequence[int],
        anchor_idx: int,
    ) -> str:
        slab = atoms.copy()
        for idx in sorted((int(i) for i in ads_indices), reverse=True):
            del slab[idx]
        try:
            registry = build_surface_site_registry(
                slab,
                site_elements=tuple(getattr(self.config, "site_elements", ())),
                substrate_elements=tuple(getattr(self.config, "substrate_elements", ())),
                surface_side=str(getattr(self.config, "surface_side", "top")),
                site_types=tuple(getattr(self.config, "site_types", ("atop", "fcc", "hcp"))),
                layer_tol=float(getattr(self.config, "surface_layer_tol", 0.5)),
                xy_tol=float(getattr(self.config, "site_match_tol", 0.6)),
                support_xy_tol=float(getattr(self.config, "support_xy_tol", 1.2)),
                termination_site_xy_tol=getattr(
                    self.config,
                    "termination_site_xy_tol",
                    None,
                ),
                vertical_offset=float(getattr(self.config, "vertical_offset", 1.5)),
                termination_elements=tuple(
                    getattr(self.config, "functional_elements", ())
                ),
                min_termination_dist=float(
                    getattr(self.config, "termination_clearance", 0.8)
                ),
            )
        except Exception:
            registry = []

        anchor_xy = atoms.positions[anchor_idx, :2]
        best: tuple[float, dict[str, object]] | None = None
        for site in registry:
            xy = np.asarray(site.get("xy", (np.nan, np.nan)), dtype=float)
            if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
                continue
            dist = self._xy_distance(atoms, anchor_xy, xy[:2])
            if best is None or dist < best[0]:
                best = (dist, site)
        if best is not None and best[0] <= self._boltzmann_geometry_site_tol():
            support = "-".join(
                str(int(idx)) for idx in sorted(best[1].get("support_indices", []))
            )
            return f"{best[1].get('site_type', 'site')}:{support}"

        tol = max(self._boltzmann_geometry_site_tol(), 1.0e-8)
        rounded = tuple(int(round(float(value) / tol)) for value in anchor_xy[:2])
        return f"anchor_xy:{rounded[0]}:{rounded[1]}"

    def _same_geometry_cluster(
        self,
        left: dict[str, object],
        right: dict[str, object],
    ) -> bool:
        if left["site_key"] != right["site_key"]:
            return False
        if left["symbols"] != right["symbols"]:
            return False
        rel_left = np.asarray(left["relative_positions"], dtype=float)
        rel_right = np.asarray(right["relative_positions"], dtype=float)
        if rel_left.shape != rel_right.shape:
            return False
        rmsd = float(np.sqrt(np.mean(np.sum((rel_left - rel_right) ** 2, axis=1))))
        if rmsd > self._boltzmann_geometry_rmsd_tol():
            return False

        return self._same_local_environment(
            left.get("local_environment"),
            right.get("local_environment"),
        )

    def _geometry_descriptor_distance(
        self,
        left: dict[str, object],
        right: dict[str, object],
    ) -> float:
        if left.get("symbols") != right.get("symbols"):
            return float("inf")
        rel_left = np.asarray(left.get("relative_positions", ()), dtype=float)
        rel_right = np.asarray(right.get("relative_positions", ()), dtype=float)
        if rel_left.shape != rel_right.shape:
            return float("inf")
        distance = float(np.sqrt(np.mean(np.sum((rel_left - rel_right) ** 2, axis=1))))
        if left.get("site_key") != right.get("site_key"):
            distance += 10.0
        env_distance = self._local_environment_distance(
            left.get("local_environment"),
            right.get("local_environment"),
        )
        if math.isfinite(env_distance):
            distance += env_distance
        return distance

    def _same_local_environment(
        self,
        left: object,
        right: object,
    ) -> bool:
        if left is None and right is None:
            return True
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if left.get("symbols") != right.get("symbols"):
            return False
        rel_left = np.asarray(left.get("relative_positions", ()), dtype=float)
        rel_right = np.asarray(right.get("relative_positions", ()), dtype=float)
        if rel_left.shape != rel_right.shape:
            return False
        if rel_left.size == 0:
            return True
        rmsd = float(np.sqrt(np.mean(np.sum((rel_left - rel_right) ** 2, axis=1))))
        return rmsd <= self._basin_local_env_rmsd_tol()

    @staticmethod
    def _local_environment_distance(
        left: object,
        right: object,
    ) -> float:
        if left is None and right is None:
            return 0.0
        if not isinstance(left, dict) or not isinstance(right, dict):
            return float("inf")
        if left.get("symbols") != right.get("symbols"):
            return float("inf")
        rel_left = np.asarray(left.get("relative_positions", ()), dtype=float)
        rel_right = np.asarray(right.get("relative_positions", ()), dtype=float)
        if rel_left.shape != rel_right.shape:
            return float("inf")
        if rel_left.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.sum((rel_left - rel_right) ** 2, axis=1))))

    def _logsumexp_free_energy(self, energies: Sequence[float]) -> float:
        temperature = self._boltzmann_temperature()
        kbt = _KB_EV_PER_K * temperature
        minimum = min(energies)
        partition = sum(math.exp(-(energy - minimum) / kbt) for energy in energies)
        return minimum - kbt * math.log(partition)

    def _che_row(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        selected: dict[str, object],
        *,
        electronic_only: bool = False,
    ) -> dict[str, object]:
        states = selected["states"]
        clean = self._g_state(states["clean"], "clean")
        oh = self._g_state(states["oh"], "oh")
        o = self._g_state(states["o"], "o")
        ooh = self._g_state(states["ooh"], "ooh")
        h2 = self._g_reference("h2", electronic_only=electronic_only)
        h2o = self._g_reference("h2o", electronic_only=electronic_only)

        dg1 = oh - clean - h2o + 0.5 * h2
        dg2 = o - oh + 0.5 * h2
        dg3 = ooh - o - h2o + 0.5 * h2
        dg4 = self._delta_g4(
            clean=clean,
            ooh=ooh,
            h2=h2,
            dg1=dg1,
            dg2=dg2,
            dg3=dg3,
            electronic_only=electronic_only,
        )
        deltas = [dg1, dg2, dg3, dg4]
        limiting_index, limiting_value = max(enumerate(deltas, start=1), key=lambda item: item[1])
        equilibrium = float(self.che_config.get("equilibrium_potential_V", 1.23))
        potential = float(self.che_config.get("potential_V", 0.0))
        shifted = [value - potential for value in deltas]
        limiting_at_u = max(shifted)

        site_manifest = next(iter(by_state.values()), {})
        return {
            "site_id": site_id,
            "site_population_rank": site_manifest.get("site_population_rank", ""),
            "population_total": self._site_population(site_manifest),
            "ready": True,
            "clean_reference_site_id": states["clean"].get("_source_site_id", site_id),
            "DeltaG1_eV": dg1,
            "DeltaG2_eV": dg2,
            "DeltaG3_eV": dg3,
            "DeltaG4_eV": dg4,
            "limiting_step": int(limiting_index),
            "limiting_DeltaG_eV": limiting_value,
            "overpotential_V": limiting_value - equilibrium,
            "potential_V": potential,
            "limiting_DeltaG_at_U_eV": limiting_at_u,
            "missing_states": "",
        }

    def _delta_g4(
        self,
        *,
        clean: float,
        ooh: float,
        h2: float,
        dg1: float,
        dg2: float,
        dg3: float,
        electronic_only: bool = False,
    ) -> float:
        if self._oer_reference_mode() == "explicit_o2":
            o2 = self._g_reference("o2", electronic_only=electronic_only)
            return clean + o2 - ooh + 0.5 * h2
        total = float(self.che_config.get("total_oer_free_energy_eV", 4.92))
        return total - dg1 - dg2 - dg3

    def _boltzmann_che_row(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        selected: dict[str, object],
        *,
        electronic_only: bool = False,
    ) -> dict[str, object]:
        base = self._che_row(
            site_id,
            by_state,
            selected,
            electronic_only=electronic_only,
        )
        counts = selected.get("state_counts", {})
        return {
            "site_id": base["site_id"],
            "site_population_rank": base["site_population_rank"],
            "population_total": base["population_total"],
            "ready": base["ready"],
            "clean_reference_site_id": base["clean_reference_site_id"],
            "boltzmann_temperature_K": self._boltzmann_temperature(),
            "boltzmann_energy_cluster_tol_eV": self._boltzmann_energy_cluster_tol(),
            "n_OH_states": counts.get("oh", ""),
            "n_O_states": counts.get("o", ""),
            "n_OOH_states": counts.get("ooh", ""),
            "DeltaG1_eV": base["DeltaG1_eV"],
            "DeltaG2_eV": base["DeltaG2_eV"],
            "DeltaG3_eV": base["DeltaG3_eV"],
            "DeltaG4_eV": base["DeltaG4_eV"],
            "limiting_step": base["limiting_step"],
            "limiting_DeltaG_eV": base["limiting_DeltaG_eV"],
            "overpotential_V": base["overpotential_V"],
            "potential_V": base["potential_V"],
            "limiting_DeltaG_at_U_eV": base["limiting_DeltaG_at_U_eV"],
            "missing_states": base["missing_states"],
        }

    def _incomplete_row(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        selected: dict[str, object],
    ) -> dict[str, object]:
        site_manifest = next(iter(by_state.values()), {})
        return {
            "site_id": site_id,
            "site_population_rank": site_manifest.get("site_population_rank", ""),
            "population_total": self._site_population(site_manifest),
            "ready": False,
            "clean_reference_site_id": "",
            "DeltaG1_eV": "",
            "DeltaG2_eV": "",
            "DeltaG3_eV": "",
            "DeltaG4_eV": "",
            "limiting_step": "",
            "limiting_DeltaG_eV": "",
            "overpotential_V": "",
            "potential_V": self.che_config.get("potential_V", 0.0),
            "limiting_DeltaG_at_U_eV": "",
            "missing_states": " ".join(selected.get("missing_states", [])),
        }

    def _incomplete_boltzmann_row(
        self,
        site_id: str,
        by_state: dict[str, dict[str, object]],
        selected: dict[str, object],
    ) -> dict[str, object]:
        base = self._incomplete_row(site_id, by_state, selected)
        return {
            "site_id": base["site_id"],
            "site_population_rank": base["site_population_rank"],
            "population_total": base["population_total"],
            "ready": False,
            "clean_reference_site_id": base["clean_reference_site_id"],
            "boltzmann_temperature_K": self._boltzmann_temperature(),
            "boltzmann_energy_cluster_tol_eV": self._boltzmann_energy_cluster_tol(),
            "n_OH_states": "",
            "n_O_states": "",
            "n_OOH_states": "",
            "DeltaG1_eV": "",
            "DeltaG2_eV": "",
            "DeltaG3_eV": "",
            "DeltaG4_eV": "",
            "limiting_step": "",
            "limiting_DeltaG_eV": "",
            "overpotential_V": "",
            "potential_V": base["potential_V"],
            "limiting_DeltaG_at_U_eV": "",
            "missing_states": base["missing_states"],
        }

    def _explicit_route_output_row(
        self,
        base: dict[str, object],
        oh_state: dict[str, object],
        o_state: dict[str, object],
        ooh_state: dict[str, object],
        *,
        o_index: int,
        ooh_index: int,
        parent_weight: float,
        o_probability: float,
        ooh_probability: float,
        n_o_basins: int,
        n_ooh_basins: int,
    ) -> dict[str, object]:
        route_weight_raw = parent_weight * o_probability * ooh_probability
        route_id = (
            f"{base.get('site_id', '')}|"
            f"O{o_index:03d}|OOH{ooh_index:03d}"
        )
        return {
            "site_id": base.get("site_id", ""),
            "route_id": route_id,
            "route_mode": "explicit_basin",
            "site_population_rank": base.get("site_population_rank", ""),
            "population_total": base.get("population_total", ""),
            "clean_reference_site_id": base.get("clean_reference_site_id", ""),
            "ready": base.get("ready", False),
            "parent_weight": parent_weight,
            "o_basin_probability": o_probability,
            "ooh_basin_probability": ooh_probability,
            "route_weight_raw": route_weight_raw,
            "route_weight": "",
            "ready_min": base.get("ready", False),
            "oh_candidate_id": oh_state.get("candidate_id", ""),
            "o_candidate_id": o_state.get("candidate_id", ""),
            "o_basin_id": o_state.get("basin_id", ""),
            "o_basin_count": o_state.get("basin_count", ""),
            "o_basin_selected_count": o_state.get("basin_selected_count", ""),
            "o_basin_trajectory_count": o_state.get("basin_trajectory_count", ""),
            "o_basin_trajectory_total_count": o_state.get(
                "basin_trajectory_total_count",
                "",
            ),
            "o_basin_trajectory_assigned_count": o_state.get(
                "basin_trajectory_assigned_count",
                "",
            ),
            "o_basin_weight_source": o_state.get("basin_weight_source", "selected"),
            "n_O_candidates": n_o_basins,
            "o_basin_candidate_ids": o_state.get("basin_candidate_ids", ""),
            "ooh_candidate_id": ooh_state.get("candidate_id", ""),
            "ooh_basin_id": ooh_state.get("basin_id", ""),
            "ooh_basin_count": ooh_state.get("basin_count", ""),
            "ooh_basin_selected_count": ooh_state.get("basin_selected_count", ""),
            "ooh_basin_trajectory_count": ooh_state.get(
                "basin_trajectory_count",
                "",
            ),
            "ooh_basin_trajectory_total_count": ooh_state.get(
                "basin_trajectory_total_count",
                "",
            ),
            "ooh_basin_trajectory_assigned_count": ooh_state.get(
                "basin_trajectory_assigned_count",
                "",
            ),
            "ooh_basin_weight_source": ooh_state.get(
                "basin_weight_source",
                "selected",
            ),
            "n_OOH_candidates": n_ooh_basins,
            "ooh_basin_candidate_ids": ooh_state.get("basin_candidate_ids", ""),
            "parent_o_candidate_id": ooh_state.get("parent_o_candidate_id", ""),
            "parent_state_dir": ooh_state.get("parent_state_dir", ""),
            "parent_state_candidate_id": ooh_state.get(
                "parent_state_candidate_id",
                "",
            ),
            "parent_candidate_id": ooh_state.get("parent_candidate_id", ""),
            "transition_builder": ooh_state.get("transition_builder", ""),
            "DeltaG1_eV": base.get("DeltaG1_eV", ""),
            "DeltaG2_eV": base.get("DeltaG2_eV", ""),
            "DeltaG3_eV": base.get("DeltaG3_eV", ""),
            "DeltaG4_eV": base.get("DeltaG4_eV", ""),
            "DeltaG1_min_eV": base.get("DeltaG1_eV", ""),
            "DeltaG2_min_eV": base.get("DeltaG2_eV", ""),
            "DeltaG3_min_eV": base.get("DeltaG3_eV", ""),
            "DeltaG4_min_eV": base.get("DeltaG4_eV", ""),
            "limiting_step": base.get("limiting_step", ""),
            "limiting_DeltaG_eV": base.get("limiting_DeltaG_eV", ""),
            "overpotential_V": base.get("overpotential_V", ""),
            "limiting_step_min": base.get("limiting_step", ""),
            "limiting_DeltaG_min_eV": base.get("limiting_DeltaG_eV", ""),
            "overpotential_min_V": base.get("overpotential_V", ""),
            "potential_V": base.get("potential_V", ""),
            "limiting_DeltaG_at_U_eV": base.get("limiting_DeltaG_at_U_eV", ""),
            "limiting_DeltaG_min_at_U_eV": base.get("limiting_DeltaG_at_U_eV", ""),
            "missing_states": base.get("missing_states", ""),
            "missing_states_min": base.get("missing_states", ""),
        }

    def _normalize_route_weights(self, rows: Sequence[dict[str, object]]) -> None:
        total = sum(
            max(0.0, self._as_float(row.get("route_weight_raw")))
            for row in rows
            if self._as_bool(row.get("ready", row.get("ready_min", False)))
        )
        if total <= 0.0:
            ready = [
                row
                for row in rows
                if self._as_bool(row.get("ready", row.get("ready_min", False)))
            ]
            if not ready:
                return
            uniform = 1.0 / len(ready)
            for row in ready:
                row["route_weight"] = uniform
            return
        for row in rows:
            raw = max(0.0, self._as_float(row.get("route_weight_raw")))
            row["route_weight"] = raw / total if raw > 0.0 else 0.0

    def _route_output_row(
        self,
        min_row: dict[str, object],
        boltzmann_row: dict[str, object] | None,
        min_electronic_row: dict[str, object] | None,
        boltzmann_electronic_row: dict[str, object] | None,
    ) -> dict[str, object]:
        route = {
            "site_id": min_row.get("site_id", ""),
            "site_population_rank": min_row.get("site_population_rank", ""),
            "population_total": min_row.get("population_total", ""),
            "clean_reference_site_id": min_row.get("clean_reference_site_id", ""),
            "ready_min": min_row.get("ready", False),
            "DeltaG1_min_eV": min_row.get("DeltaG1_eV", ""),
            "DeltaG2_min_eV": min_row.get("DeltaG2_eV", ""),
            "DeltaG3_min_eV": min_row.get("DeltaG3_eV", ""),
            "DeltaG4_min_eV": min_row.get("DeltaG4_eV", ""),
            "ready_min_electronic": "",
            "DeltaG1_min_electronic_eV": "",
            "DeltaG2_min_electronic_eV": "",
            "DeltaG3_min_electronic_eV": "",
            "DeltaG4_min_electronic_eV": "",
            "limiting_step_min_electronic": "",
            "limiting_DeltaG_min_electronic_eV": "",
            "overpotential_min_electronic_V": "",
            "limiting_step_min": min_row.get("limiting_step", ""),
            "limiting_DeltaG_min_eV": min_row.get("limiting_DeltaG_eV", ""),
            "overpotential_min_V": min_row.get("overpotential_V", ""),
            "missing_states_min": min_row.get("missing_states", ""),
            "missing_states_min_electronic": "",
            "ready_boltzmann": "",
            "boltzmann_temperature_K": "",
            "boltzmann_energy_cluster_tol_eV": "",
            "n_OH_candidates": "",
            "n_O_candidates": "",
            "n_OOH_candidates": "",
            "DeltaG1_boltzmann_eV": "",
            "DeltaG2_boltzmann_eV": "",
            "DeltaG3_boltzmann_eV": "",
            "DeltaG4_boltzmann_eV": "",
            "ready_boltzmann_electronic": "",
            "DeltaG1_boltzmann_electronic_eV": "",
            "DeltaG2_boltzmann_electronic_eV": "",
            "DeltaG3_boltzmann_electronic_eV": "",
            "DeltaG4_boltzmann_electronic_eV": "",
            "limiting_step_boltzmann_electronic": "",
            "limiting_DeltaG_boltzmann_electronic_eV": "",
            "overpotential_boltzmann_electronic_V": "",
            "limiting_step_boltzmann": "",
            "limiting_DeltaG_boltzmann_eV": "",
            "overpotential_boltzmann_V": "",
            "missing_states_boltzmann": "",
            "missing_states_boltzmann_electronic": "",
            "potential_V": min_row.get("potential_V", ""),
            "limiting_DeltaG_min_at_U_eV": min_row.get(
                "limiting_DeltaG_at_U_eV",
                "",
            ),
            "limiting_DeltaG_min_electronic_at_U_eV": "",
            "limiting_DeltaG_boltzmann_at_U_eV": "",
            "limiting_DeltaG_boltzmann_electronic_at_U_eV": "",
        }
        if min_electronic_row is not None:
            self._fill_route_electronic_columns(
                route,
                "min",
                min_electronic_row,
            )
        if boltzmann_row is None:
            return route

        route["clean_reference_site_id"] = (
            route["clean_reference_site_id"]
            or boltzmann_row.get("clean_reference_site_id", "")
        )
        route.update(
            {
                "ready_boltzmann": boltzmann_row.get("ready", False),
                "boltzmann_temperature_K": boltzmann_row.get(
                    "boltzmann_temperature_K",
                    "",
                ),
                "boltzmann_energy_cluster_tol_eV": boltzmann_row.get(
                    "boltzmann_energy_cluster_tol_eV",
                    "",
                ),
                "n_OH_candidates": boltzmann_row.get("n_OH_states", ""),
                "n_O_candidates": boltzmann_row.get("n_O_states", ""),
                "n_OOH_candidates": boltzmann_row.get("n_OOH_states", ""),
                "DeltaG1_boltzmann_eV": boltzmann_row.get("DeltaG1_eV", ""),
                "DeltaG2_boltzmann_eV": boltzmann_row.get("DeltaG2_eV", ""),
                "DeltaG3_boltzmann_eV": boltzmann_row.get("DeltaG3_eV", ""),
                "DeltaG4_boltzmann_eV": boltzmann_row.get("DeltaG4_eV", ""),
                "limiting_step_boltzmann": boltzmann_row.get("limiting_step", ""),
                "limiting_DeltaG_boltzmann_eV": boltzmann_row.get(
                    "limiting_DeltaG_eV",
                    "",
                ),
                "overpotential_boltzmann_V": boltzmann_row.get(
                    "overpotential_V",
                    "",
                ),
                "missing_states_boltzmann": boltzmann_row.get("missing_states", ""),
                "limiting_DeltaG_boltzmann_at_U_eV": boltzmann_row.get(
                    "limiting_DeltaG_at_U_eV",
                    "",
                ),
            }
        )
        if boltzmann_electronic_row is not None:
            self._fill_route_electronic_columns(
                route,
                "boltzmann",
                boltzmann_electronic_row,
            )
        return route

    @staticmethod
    def _fill_route_electronic_columns(
        route: dict[str, object],
        prefix: str,
        row: dict[str, object],
    ) -> None:
        route[f"ready_{prefix}_electronic"] = row.get("ready", False)
        route[f"DeltaG1_{prefix}_electronic_eV"] = row.get("DeltaG1_eV", "")
        route[f"DeltaG2_{prefix}_electronic_eV"] = row.get("DeltaG2_eV", "")
        route[f"DeltaG3_{prefix}_electronic_eV"] = row.get("DeltaG3_eV", "")
        route[f"DeltaG4_{prefix}_electronic_eV"] = row.get("DeltaG4_eV", "")
        route[f"limiting_step_{prefix}_electronic"] = row.get("limiting_step", "")
        route[f"limiting_DeltaG_{prefix}_electronic_eV"] = row.get(
            "limiting_DeltaG_eV",
            "",
        )
        route[f"overpotential_{prefix}_electronic_V"] = row.get(
            "overpotential_V",
            "",
        )
        route[f"missing_states_{prefix}_electronic"] = row.get(
            "missing_states",
            "",
        )
        route[f"limiting_DeltaG_{prefix}_electronic_at_U_eV"] = row.get(
            "limiting_DeltaG_at_U_eV",
            "",
        )

    def _state_output_rows(
        self,
        site_id: str,
        selected: dict[str, object],
        weighted: dict[str, object] | None,
        selected_electronic: dict[str, object] | None = None,
        weighted_electronic: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        min_states = selected.get("states", {})
        weighted_states = (weighted or {}).get("states", {})
        weighted_counts = (weighted or {}).get("state_counts", {})
        min_electronic_states = (selected_electronic or {}).get("states", {})
        weighted_electronic_states = (weighted_electronic or {}).get("states", {})
        weighted_electronic_counts = (weighted_electronic or {}).get("state_counts", {})

        for label in ("clean", "oh", "o", "ooh"):
            min_state = min_states.get(label)
            weighted_state = weighted_states.get(label)
            min_electronic_state = min_electronic_states.get(label)
            weighted_electronic_state = weighted_electronic_states.get(label)
            if (
                min_state is None
                and weighted_state is None
                and min_electronic_state is None
                and weighted_electronic_state is None
            ):
                rows.append(self._missing_state_output_row(site_id, label))
                continue

            reference = (
                min_state
                or weighted_state
                or min_electronic_state
                or weighted_electronic_state
                or {}
            )
            rows.append(
                {
                    "site_id": site_id,
                    "state_label": label,
                    "ready": True,
                    "missing": False,
                    "source_site_id": reference.get("_source_site_id", site_id),
                    "source_state_dir": reference.get("_source_state_dir", ""),
                    "selection_mode": reference.get("_reference_mode", ""),
                    "min_candidate_id": reference.get("candidate_id", ""),
                    "min_candidate_kind": reference.get("candidate_kind", ""),
                    "parent_o_candidate_id": reference.get(
                        "parent_o_candidate_id",
                        "",
                    ),
                    "parent_state_dir": reference.get("parent_state_dir", ""),
                    "parent_state_candidate_id": reference.get(
                        "parent_state_candidate_id",
                        "",
                    ),
                    "parent_candidate_id": reference.get("parent_candidate_id", ""),
                    "transition_builder": reference.get("transition_builder", ""),
                    "min_free_energy_eV": (
                        min_state.get("energy_eV", "") if min_state else ""
                    ),
                    "min_energy_source": reference.get("energy_source", "electronic"),
                    "electronic_energy_eV": reference.get("electronic_energy_eV", ""),
                    "harmonic_correction_eV": reference.get(
                        "harmonic_correction_eV",
                        "",
                    ),
                    "converged": reference.get("converged", ""),
                    "fmax_eV_A": reference.get("fmax_eV_A", ""),
                    "nsteps": reference.get("nsteps", ""),
                    "used_unconverged_fallback": reference.get(
                        "used_unconverged_fallback",
                        "",
                    ),
                    "boltzmann_free_energy_eV": (
                        weighted_state.get("energy_eV", "") if weighted_state else ""
                    ),
                    "n_boltzmann_candidates": weighted_counts.get(label, ""),
                    "min_electronic_free_energy_eV": (
                        min_electronic_state.get("energy_eV", "")
                        if min_electronic_state
                        else ""
                    ),
                    "min_electronic_candidate_id": (
                        min_electronic_state.get("candidate_id", "")
                        if min_electronic_state
                        else ""
                    ),
                    "boltzmann_electronic_free_energy_eV": (
                        weighted_electronic_state.get("energy_eV", "")
                        if weighted_electronic_state
                        else ""
                    ),
                    "n_boltzmann_electronic_candidates": (
                        weighted_electronic_counts.get(label, "")
                    ),
                    "boltzmann_temperature_K": (
                        self._boltzmann_temperature()
                        if weighted_state is not None
                        else ""
                    ),
                    "boltzmann_energy_cluster_tol_eV": (
                        self._boltzmann_energy_cluster_tol()
                        if weighted_state is not None
                        else ""
                    ),
                    "energies_csv": reference.get("energies_csv", ""),
                }
            )
        return rows

    @staticmethod
    def _missing_state_output_row(site_id: str, label: str) -> dict[str, object]:
        return {
            "site_id": site_id,
            "state_label": label,
            "ready": False,
            "missing": True,
            "source_site_id": "",
            "source_state_dir": "",
            "selection_mode": "",
            "min_candidate_id": "",
            "min_candidate_kind": "",
            "parent_o_candidate_id": "",
            "parent_state_dir": "",
            "parent_state_candidate_id": "",
            "parent_candidate_id": "",
            "transition_builder": "",
            "min_free_energy_eV": "",
            "min_energy_source": "",
            "electronic_energy_eV": "",
            "harmonic_correction_eV": "",
            "converged": "",
            "fmax_eV_A": "",
            "nsteps": "",
            "used_unconverged_fallback": "",
            "boltzmann_free_energy_eV": "",
            "n_boltzmann_candidates": "",
            "min_electronic_free_energy_eV": "",
            "min_electronic_candidate_id": "",
            "boltzmann_electronic_free_energy_eV": "",
            "n_boltzmann_electronic_candidates": "",
            "boltzmann_temperature_K": "",
            "boltzmann_energy_cluster_tol_eV": "",
            "energies_csv": "",
        }

    def _ensemble_summary_rows(
        self,
        rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        ready = [
            row
            for row in rows
            if self._as_bool(row.get("ready", row.get("ready_min", False)))
            and math.isfinite(
                self._as_float(row.get("overpotential_V", row.get("overpotential_min_V")))
            )
        ]
        if not ready:
            return []

        population_sum = sum(
            max(0.0, self._as_float(row.get("route_weight_raw")))
            for row in ready
        )
        if population_sum > 0.0:
            population_missing = max(0.0, 1.0 - population_sum)
        else:
            population_missing = 0.0

        weights = [self._as_float(row.get("route_weight")) for row in ready]
        if any(not math.isfinite(value) for value in weights) or sum(weights) <= 0.0:
            weights = [1.0 / len(ready)] * len(ready)
        eta_weighted_mean = sum(
            weight * self._as_float(row.get("overpotential_V", row.get("overpotential_min_V")))
            for weight, row in zip(weights, ready)
        )
        min_eta_row = min(
            ready,
            key=lambda row: self._as_float(row.get("overpotential_V", row.get("overpotential_min_V"))),
        )
        dominant_weight, dominant_row = max(
            zip(weights, ready),
            key=lambda item: item[0],
        )

        return [
            {
                "source": "oer_routes.csv",
                "n_ready_routes": len(ready),
                "population_sum_ready": population_sum,
                "population_missing": population_missing,
                "route_weight_model": _ROUTE_WEIGHT_MODEL,
                "basin_cluster_mode": self._basin_cluster_mode(),
                "basin_energy_cluster_tol_eV": self._basin_energy_cluster_tol(),
                "route_weighted_mean_overpotential_V": eta_weighted_mean,
                "min_site_overpotential_V": self._as_float(
                    min_eta_row.get("overpotential_V", min_eta_row.get("overpotential_min_V")),
                ),
                "min_overpotential_route_id": min_eta_row.get("route_id", ""),
                "min_overpotential_site_id": min_eta_row.get("site_id", ""),
                "dominant_route_id": dominant_row.get("route_id", ""),
                "dominant_route_site_id": dominant_row.get("site_id", ""),
                "dominant_route_weight": dominant_weight,
            }
        ]

    def _g_state(self, row: dict[str, object], label: str) -> float:
        correction = float(self.che_config.get(f"{label}_correction_eV", 0.0))
        return self._as_float(row.get("energy_eV")) + correction

    def _g_reference(self, label: str, *, electronic_only: bool = False) -> float:
        cache_key = f"{label}:electronic" if electronic_only else label
        cached = self._reference_cache.get(cache_key)
        if cached is not None:
            return cached

        energy = float(self.che_config[f"{label}_energy_eV"])
        value = energy
        thermo_cfg = self._reference_thermo_config(label)
        if (not electronic_only) and bool(thermo_cfg.get("enabled", False)):
            vib_energies = [
                float(item) for item in (thermo_cfg.get("vib_energies_eV") or [])
            ]
            geometry = str(thermo_cfg.get("geometry") or "").strip()
            if not geometry:
                raise ValueError(f"che.{label}_thermo.geometry is required when enabled.")
            symmetrynumber = thermo_cfg.get("symmetrynumber")
            spin = thermo_cfg.get("spin")
            temperature = thermo_cfg.get("temperature_K")
            if temperature in (None, ""):
                temperature = self._boltzmann_temperature()
            pressure = thermo_cfg.get("pressure_Pa", 101325.0)
            thermo = IdealGasThermo(
                vib_energies=vib_energies,
                geometry=geometry,
                potentialenergy=energy,
                atoms=self._reference_atoms(label, thermo_cfg),
                symmetrynumber=None if symmetrynumber in (None, "") else int(symmetrynumber),
                spin=None if spin in (None, "") else float(spin),
                ignore_imag_modes=bool(thermo_cfg.get("ignore_imag_modes", False)),
            )
            value = float(
                thermo.get_gibbs_energy(
                    float(temperature),
                    float(pressure),
                    verbose=False,
                )
            )
        correction = float(self.che_config.get(f"{label}_correction_eV", 0.0))
        value += correction
        self._reference_cache[cache_key] = value
        return value

    def _reference_thermo_config(self, label: str) -> dict[str, object]:
        raw = self.che_config.get(f"{label}_thermo", {}) or {}
        merged = dict(_DEFAULT_REFERENCE_THERMO)
        if isinstance(raw, dict):
            merged.update(raw)
        return merged

    def _reference_atoms(self, label: str, thermo_cfg: dict[str, object]):
        value = thermo_cfg.get("atoms")
        if value not in (None, ""):
            from ase.io import read

            return read(str(value))
        aliases = {
            "h2": "H2",
            "h2o": "H2O",
            "o2": "O2",
        }
        name = aliases.get(label)
        if name is None:
            raise ValueError(
                f"che.{label}_thermo.atoms is required when no built-in reference is available."
            )
        return molecule(name)

    @staticmethod
    def _read_atoms_list(path: Path):
        atoms = read(str(path), index=":")
        if isinstance(atoms, list):
            return atoms
        return [atoms]

    @staticmethod
    def _candidate_index(row: dict[str, object]) -> int:
        try:
            return int(float(str(row.get("candidate_index", ""))))
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _adsorbate_indices(atoms, candidate_row: dict[str, object]) -> list[int]:
        if "reaction_is_adsorbate" in atoms.arrays:
            mask = np.asarray(atoms.arrays["reaction_is_adsorbate"], dtype=bool)
            return [int(idx) for idx, value in enumerate(mask) if value]

        text = str(candidate_row.get("atom_is_adsorbate", "")).strip()
        if text:
            try:
                values = [int(part) for part in text.split()]
            except ValueError:
                values = []
            if len(values) == len(atoms):
                return [idx for idx, value in enumerate(values) if value]

        tags = atoms.get_tags()
        if len(tags):
            return [int(idx) for idx, value in enumerate(tags) if int(value) != 0]
        return []

    @staticmethod
    def _xy_distance(atoms, xy_a: np.ndarray, xy_b: np.ndarray) -> float:
        point_a = np.array([float(xy_a[0]), float(xy_a[1]), 0.0], dtype=float)
        point_b = np.array([float(xy_b[0]), float(xy_b[1]), 0.0], dtype=float)
        if any(atoms.pbc[:2]):
            delta = get_distances(
                point_a.reshape(1, 3),
                point_b.reshape(1, 3),
                cell=atoms.get_cell(),
                pbc=atoms.get_pbc(),
            )[0][0, 0]
            return float(np.linalg.norm(delta[:2]))
        return float(np.linalg.norm((point_b - point_a)[:2]))

    @staticmethod
    def _site_population(manifest_row: dict[str, object]) -> object:
        value = manifest_row.get("population_total", "")
        if value not in (None, ""):
            return value
        site_dir = Path(str(manifest_row.get("site_dir", "")))
        metadata = site_dir / "representatives.csv"
        if not metadata.exists():
            return ""
        rows = OERCHESummarizer._read_csv(metadata)
        if not rows:
            return ""
        return rows[0].get("population_total", "")

    @staticmethod
    def _as_bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _row_bool_or_true(value: object) -> bool:
        if value in (None, ""):
            return True
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _as_float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _read_csv(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
