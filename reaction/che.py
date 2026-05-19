"""CHE post-processing for site-conditioned OER states."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from ase.build import molecule
from ase.thermochemistry import IdealGasThermo

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
    "boltzmann_weight_states": False,
    "boltzmann_temperature_K": 303.0,
    "boltzmann_energy_cluster_tol_eV": 0.0,
}

_KB_EV_PER_K = 8.617333262145e-5


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
            weighted: dict[str, object] | None = None
            weighted_electronic: dict[str, object] | None = None
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
            if self._boltzmann_enabled():
                weighted = self._select_site_boltzmann_states(
                    site_id,
                    by_state,
                    shared_clean,
                )
                weighted_electronic = self._select_site_electronic_states(
                    site_id,
                    by_state,
                    shared_clean,
                    boltzmann=True,
                )
                if weighted["ready"]:
                    boltzmann_row = self._boltzmann_che_row(site_id, by_state, weighted)
                else:
                    boltzmann_row = self._incomplete_boltzmann_row(
                        site_id,
                        by_state,
                        weighted,
                    )
                if weighted_electronic["ready"]:
                    boltzmann_electronic_row = self._boltzmann_che_row(
                        site_id,
                        by_state,
                        weighted_electronic,
                        electronic_only=True,
                    )
                else:
                    boltzmann_electronic_row = self._incomplete_boltzmann_row(
                        site_id,
                        by_state,
                        weighted_electronic,
                    )
            else:
                boltzmann_row = None
                boltzmann_electronic_row = None
            route_rows.append(
                self._route_output_row(
                    min_row,
                    boltzmann_row,
                    min_electronic_row,
                    boltzmann_electronic_row,
                )
            )
            state_rows.extend(
                self._state_output_rows(
                    site_id,
                    selected,
                    weighted,
                    selected_electronic,
                    weighted_electronic,
                )
            )

        routes_path = output_path(self.config, "oer_routes_csv")
        write_csv(routes_path, route_rows)
        states_path = output_path(self.config, "oer_states_csv")
        write_csv(states_path, state_rows)
        outputs = {
            "oer_routes_csv": str(routes_path),
            "oer_states_csv": str(states_path),
        }
        if self._boltzmann_enabled():
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
        temperature = float(self.che_config.get("boltzmann_temperature_K", 303.0))
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("che.boltzmann_temperature_K must be a positive number.")
        return temperature

    def _boltzmann_energy_cluster_tol(self) -> float:
        tol = float(self.che_config.get("boltzmann_energy_cluster_tol_eV", 0.0))
        if tol < 0.0 or not math.isfinite(tol):
            raise ValueError(
                "che.boltzmann_energy_cluster_tol_eV must be a non-negative number."
            )
        return tol

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
        clustered = self._cluster_energy_rows(finite)
        energies = [self._as_float(row.get("energy_eV")) for row in clustered]
        free_energy = self._logsumexp_free_energy(energies)
        best = min(clustered, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energy_eV"] = free_energy
        output["energies_csv"] = str(energies_csv)
        output["n_boltzmann_states"] = len(clustered)
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
        clustered = self._cluster_energy_rows(finite)
        energies = [self._as_float(row.get("energy_eV")) for row in clustered]
        free_energy = self._logsumexp_free_energy(energies)
        best = min(clustered, key=lambda row: self._as_float(row.get("energy_eV")))
        output = dict(best)
        output["energy_eV"] = free_energy
        output["energies_csv"] = str(energies_csv)
        output["n_boltzmann_states"] = len(clustered)
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
            if self._as_bool(row.get("ready_boltzmann"))
            and math.isfinite(self._as_float(row.get("overpotential_boltzmann_V")))
        ]
        if not ready:
            return []

        population_sum = sum(
            max(0.0, self._as_float(row.get("population_total"))) for row in ready
        )
        if population_sum > 0.0:
            population_missing = max(0.0, 1.0 - population_sum)
        else:
            population_missing = 0.0

        temperature = self._boltzmann_temperature()
        kbt = _KB_EV_PER_K * temperature
        delta_g1 = [self._as_float(row.get("DeltaG1_boltzmann_eV")) for row in ready]
        minimum = min(delta_g1)
        raw_weights = [
            math.exp(-(value - minimum) / kbt)
            for value in delta_g1
        ]
        weight_sum = sum(raw_weights)
        if weight_sum > 0.0:
            weights = [value / weight_sum for value in raw_weights]
        else:
            weights = [1.0 / len(ready)] * len(ready)
        eta_weighted_mean = sum(
            weight * self._as_float(row.get("overpotential_boltzmann_V"))
            for weight, row in zip(weights, ready)
        )
        min_eta_row = min(
            ready,
            key=lambda row: self._as_float(row.get("overpotential_boltzmann_V")),
        )
        dominant_weight, dominant_row = max(
            zip(weights, ready),
            key=lambda item: item[0],
        )

        return [
            {
                "source": "oer_routes.csv",
                "n_ready_sites": len(ready),
                "population_sum_ready": population_sum,
                "population_missing": population_missing,
                "site_weight_model": "dilute_oh_deltag1",
                "site_degeneracy_model": "uniform",
                "boltzmann_temperature_K": ready[0].get("boltzmann_temperature_K", ""),
                "boltzmann_energy_cluster_tol_eV": ready[0].get(
                    "boltzmann_energy_cluster_tol_eV",
                    "",
                ),
                "route_weighted_mean_overpotential_V": eta_weighted_mean,
                "min_site_overpotential_V": self._as_float(
                    min_eta_row.get("overpotential_boltzmann_V"),
                ),
                "min_site_overpotential_site_id": min_eta_row.get("site_id", ""),
                "dominant_weight_site_id": dominant_row.get("site_id", ""),
                "dominant_weight_fraction": dominant_weight,
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
