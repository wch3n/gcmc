"""Reference molecule relaxation, vibration, and IdealGasThermo helper."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase.optimize import LBFGS

from gcmc.workflows import build_adsorbate_gcmc_calculator


_DEFAULT_REFERENCE_SPECS = {
    "H2": {
        "geometry": "linear",
        "symmetrynumber": 2,
        "spin": 0.0,
    },
    "H2O": {
        "geometry": "nonlinear",
        "symmetrynumber": 2,
        "spin": 0.0,
    },
    "O2": {
        "geometry": "linear",
        "symmetrynumber": 2,
        "spin": 1.0,
    },
}

_DEFAULT_REFERENCE_THERMO_CONFIG = {
    "reference": {
        "molecule": None,
        "atoms": None,
        "vacuum_A": 8.0,
        "temperature_K": 303.0,
        "pressure_Pa": 101325.0,
        "energy_eV": None,
        "geometry": None,
        "symmetrynumber": None,
        "spin": None,
    },
    "calculator": {
        "calculator": "lj",
        "lj_cutoff": 6.0,
        "model": None,
        "model_file": None,
        "device": "cpu",
        "use_kokkos": True,
    },
    "output": {
        "output_dir": "reference_thermo",
        "skip_existing": True,
    },
    "relaxation": {
        "enabled": True,
        "fmax": 0.001,
        "steps": 1000,
        "log_file": "relax.log",
        "progress_log": "reference_thermo.log",
        "progress_stdout": True,
    },
    "vibrations": {
        "delta_A": 0.01,
        "nfree": 2,
        "ignore_imag_modes": False,
    },
}

_DEFAULT_REFERENCE_THERMO_STAGE_CONFIG = {
    "enabled": False,
    "molecules": ["H2", "H2O"],
    "output_dir": "references",
    "skip_existing": True,
    "overwrite_che_references": False,
    "vacuum_A": 8.0,
    "temperature_K": None,
    "pressure_Pa": 101325.0,
    "calculator": {},
    "relaxation": {
        "enabled": True,
        "fmax": 0.001,
        "steps": 1000,
        "log_file": "relax.log",
        "progress_log": "reference_thermo.log",
        "progress_stdout": True,
    },
    "vibrations": {
        "delta_A": 0.01,
        "nfree": 2,
        "ignore_imag_modes": False,
    },
    "h2": {"energy_eV": None},
    "h2o": {"energy_eV": None},
    "o2": {"energy_eV": None},
}


def default_reference_thermo_config() -> dict[str, object]:
    return yaml.safe_load(yaml.safe_dump(_DEFAULT_REFERENCE_THERMO_CONFIG))


def default_reference_thermo_stage_config() -> dict[str, object]:
    return yaml.safe_load(yaml.safe_dump(_DEFAULT_REFERENCE_THERMO_STAGE_CONFIG))


def _deep_update(base: dict[str, object], updates: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_optional_path(value, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _namespace_from_reference_mapping(
    raw: dict[str, object],
    *,
    base_dir: Path | None = None,
) -> SimpleNamespace:
    merged = _deep_update(default_reference_thermo_config(), raw)
    reference = merged["reference"]
    calculator = merged["calculator"]
    output = merged["output"]
    relaxation = merged["relaxation"]
    vibrations = merged["vibrations"]

    if base_dir is not None:
        reference["atoms"] = _resolve_optional_path(reference.get("atoms"), base_dir)
        output["output_dir"] = _resolve_optional_path(output.get("output_dir"), base_dir)
        for key in ("model", "model_file"):
            calculator[key] = _resolve_optional_path(calculator.get(key), base_dir)
        for key in ("log_file", "progress_log"):
            value = relaxation.get(key)
            if value not in (None, ""):
                path = Path(str(value))
                if path.parent != Path("."):
                    relaxation[key] = _resolve_optional_path(value, base_dir)

    flat: dict[str, object] = {}
    flat.update(reference)
    flat.update(calculator)
    flat.update(output)
    flat["relaxation"] = relaxation
    flat["vibrations"] = vibrations
    return SimpleNamespace(**flat)


def load_reference_thermo_config(config_path: str | Path) -> SimpleNamespace:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Reference thermo config must be a mapping.")
    return _namespace_from_reference_mapping(raw, base_dir=config_path.parent)


class ReferenceThermoWorkflow:
    """Relax one reference molecule and compute its gas-phase Gibbs free energy."""

    def __init__(self, config: SimpleNamespace):
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ReferenceThermoWorkflow":
        return cls(load_reference_thermo_config(config_path))

    def run(self) -> dict[str, str]:
        atoms = self._build_reference_atoms()
        output_dir = Path(str(self.config.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        initial_path = output_dir / "initial.traj"
        relaxed_path = output_dir / "relaxed.traj"
        summary_path = output_dir / "summary.yaml"
        snippet_path = output_dir / "che_snippet.yaml"
        if (
            bool(getattr(self.config, "skip_existing", True))
            and summary_path.exists()
            and snippet_path.exists()
            and self._existing_summary_matches_reference(summary_path)
        ):
            return {
                "initial_traj": str(initial_path),
                "relaxed_traj": str(relaxed_path),
                "summary_yaml": str(summary_path),
                "che_snippet_yaml": str(snippet_path),
            }
        write(str(initial_path), atoms)

        calculator = self._build_calculator()
        atoms.calc = calculator

        relaxation_cfg = getattr(self.config, "relaxation", {}) or {}
        if bool(relaxation_cfg.get("enabled", True)):
            self._progress(
                "reference-thermo relax start "
                f"molecule={self._reference_name()} output_dir={output_dir}"
            )
            optimizer = LBFGS(
                atoms,
                logfile=self._relax_logfile_path(),
            )
            optimizer.run(
                fmax=float(relaxation_cfg.get("fmax", 0.001)),
                steps=int(relaxation_cfg.get("steps", 1000)),
            )
            self._progress("reference-thermo relax done")

        calculated_potential_energy = float(atoms.get_potential_energy())
        reference_energy, energy_source = self._thermo_reference_energy(
            calculated_potential_energy
        )
        if energy_source == "configured":
            self._progress(
                "reference-thermo using configured electronic energy "
                f"molecule={self._reference_name()} "
                f"energy_eV={reference_energy:.12g} "
                f"calculated_eV={calculated_potential_energy:.12g}"
            )
        atoms.calc = None
        write(str(relaxed_path), atoms)

        vibrations_cfg = getattr(self.config, "vibrations", {}) or {}
        vib_dir = output_dir / "vibrations"
        removed_cache_files = self._remove_empty_vibration_cache(vib_dir / "vib")
        if removed_cache_files:
            self._progress(
                "reference-thermo removed stale empty vibration cache files "
                f"count={removed_cache_files}"
            )
        vib_atoms = atoms.copy()
        vib_atoms.calc = calculator
        vib = Vibrations(
            vib_atoms,
            indices=list(range(len(atoms))),
            name=str(vib_dir / "vib"),
            delta=float(vibrations_cfg.get("delta_A", 0.01)),
            nfree=int(vibrations_cfg.get("nfree", 2)),
        )
        self._progress("reference-thermo vibrations start")
        vib.run()
        raw_vib_energies = list(vib.get_energies())
        self._progress("reference-thermo vibrations done")

        thermo = IdealGasThermo(
            vib_energies=raw_vib_energies,
            geometry=str(self.config.geometry),
            potentialenergy=reference_energy,
            atoms=atoms,
            symmetrynumber=int(self.config.symmetrynumber),
            spin=float(self.config.spin),
            ignore_imag_modes=bool(vibrations_cfg.get("ignore_imag_modes", False)),
        )
        temperature = float(self.config.temperature_K)
        pressure = float(self.config.pressure_Pa)
        zpe = float(thermo.get_ZPE_correction())
        enthalpy = float(thermo.get_enthalpy(temperature, verbose=False))
        entropy = float(thermo.get_entropy(temperature, pressure, verbose=False))
        gibbs = float(thermo.get_gibbs_energy(temperature, pressure, verbose=False))
        correction = getattr(self.config, "correction_eV", None)

        summary = {
            "reference_name": self._reference_name(),
            "formula": atoms.get_chemical_formula(),
            "output_dir": str(output_dir),
            "relaxed_traj": str(relaxed_path),
            "temperature_K": temperature,
            "pressure_Pa": pressure,
            "geometry": str(self.config.geometry),
            "symmetrynumber": int(self.config.symmetrynumber),
            "spin": float(self.config.spin),
            "potential_energy_eV": reference_energy,
            "reference_energy_eV": reference_energy,
            "calculated_potential_energy_eV": calculated_potential_energy,
            "energy_source": energy_source,
            "zpe_eV": zpe,
            "enthalpy_eV": enthalpy,
            "entropy_eV_K": entropy,
            "minus_TS_eV": -temperature * entropy,
            "gibbs_free_energy_eV": gibbs,
            "vibration_delta_A": float(vibrations_cfg.get("delta_A", 0.01)),
            "vibration_nfree": int(vibrations_cfg.get("nfree", 2)),
            "ignore_imag_modes": bool(vibrations_cfg.get("ignore_imag_modes", False)),
            "n_imag_modes": int(getattr(thermo, "n_imag", 0)),
            "vib_energies_eV": [float(value.real) for value in thermo.vib_energies],
        }
        if correction not in (None, ""):
            summary["correction_eV"] = float(correction)
            summary["corrected_gibbs_free_energy_eV"] = gibbs + float(correction)
        summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

        snippet = self._che_snippet(summary)
        snippet_path.write_text(yaml.safe_dump(snippet, sort_keys=False))
        return {
            "initial_traj": str(initial_path),
            "relaxed_traj": str(relaxed_path),
            "summary_yaml": str(summary_path),
            "che_snippet_yaml": str(snippet_path),
        }

    def _configured_reference_energy(self) -> float | None:
        value = getattr(self.config, "energy_eV", None)
        if value in (None, ""):
            return None
        energy = float(value)
        if not math.isfinite(energy):
            raise ValueError("reference.energy_eV must be finite when supplied.")
        return energy

    def _thermo_reference_energy(self, calculated_energy: float) -> tuple[float, str]:
        configured = self._configured_reference_energy()
        if configured is None:
            return calculated_energy, "calculator"
        return configured, "configured"

    def _existing_summary_matches_reference(self, summary_path: Path) -> bool:
        try:
            summary = yaml.safe_load(summary_path.read_text()) or {}
        except Exception:
            return False
        if not isinstance(summary, dict):
            return False

        configured = self._configured_reference_energy()
        if configured is not None:
            stored = summary.get("reference_energy_eV", summary.get("potential_energy_eV"))
            if not self._summary_float_matches(stored, configured, abs_tol=1e-10):
                return False

        for key in ("temperature_K", "pressure_Pa"):
            if not self._summary_float_matches(
                summary.get(key),
                float(getattr(self.config, key)),
                abs_tol=1e-10,
            ):
                return False
        geometry = getattr(self.config, "geometry", None)
        if geometry not in (None, "") and str(summary.get("geometry", "")) != str(geometry):
            return False
        symmetrynumber = getattr(self.config, "symmetrynumber", None)
        if symmetrynumber not in (None, ""):
            try:
                if int(summary.get("symmetrynumber", -1)) != int(symmetrynumber):
                    return False
            except (TypeError, ValueError):
                return False
        spin = getattr(self.config, "spin", None)
        if spin not in (None, ""):
            if not self._summary_float_matches(
                summary.get("spin"),
                float(spin),
                abs_tol=1e-15,
            ):
                return False
        correction = getattr(self.config, "correction_eV", None)
        if correction not in (None, ""):
            stored_correction = summary.get("correction_eV", 0.0)
            if not self._summary_float_matches(
                stored_correction,
                float(correction),
                abs_tol=1e-15,
            ):
                return False
        vibrations_cfg = getattr(self.config, "vibrations", {}) or {}
        if not self._summary_float_matches(
            summary.get("vibration_delta_A"),
            float(vibrations_cfg.get("delta_A", 0.01)),
            abs_tol=1e-15,
        ):
            return False
        try:
            if int(summary.get("vibration_nfree", -1)) != int(
                vibrations_cfg.get("nfree", 2)
            ):
                return False
        except (TypeError, ValueError):
            return False
        if bool(summary.get("ignore_imag_modes", False)) != bool(
            vibrations_cfg.get("ignore_imag_modes", False)
        ):
            return False
        return True

    @staticmethod
    def _summary_float_matches(value, expected: float, *, abs_tol: float) -> bool:
        if value in (None, ""):
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isclose(number, expected, rel_tol=0.0, abs_tol=abs_tol)

    def _build_reference_atoms(self) -> Atoms:
        if getattr(self.config, "atoms", None):
            atoms = read(str(self.config.atoms))
            if not isinstance(atoms, Atoms):
                raise ValueError("Reference atoms file must contain a single structure.")
        else:
            name = self._reference_name()
            if name is None:
                raise ValueError("reference.molecule or reference.atoms is required.")
            atoms = molecule(name)

        atoms = atoms.copy()
        atoms.pbc = False
        atoms.center(vacuum=float(getattr(self.config, "vacuum_A", 8.0)))

        defaults = _DEFAULT_REFERENCE_SPECS.get(self._reference_name() or "")
        if getattr(self.config, "geometry", None) in (None, ""):
            if defaults is None:
                raise ValueError("reference.geometry is required for custom molecules.")
            self.config.geometry = defaults["geometry"]
        if getattr(self.config, "symmetrynumber", None) in (None, ""):
            if defaults is None:
                raise ValueError("reference.symmetrynumber is required for custom molecules.")
            self.config.symmetrynumber = defaults["symmetrynumber"]
        if getattr(self.config, "spin", None) in (None, ""):
            if defaults is None:
                raise ValueError("reference.spin is required for custom molecules.")
            self.config.spin = defaults["spin"]
        return atoms

    def _reference_name(self) -> str | None:
        value = getattr(self.config, "molecule", None)
        if value in (None, ""):
            return None
        name = str(value).strip()
        aliases = {
            "h2": "H2",
            "h2o": "H2O",
            "o2": "O2",
        }
        return aliases.get(name.lower(), name)

    def _build_calculator(self):
        return build_adsorbate_gcmc_calculator(
            self.config,
            {"device": getattr(self.config, "device", None)},
        )

    def _relax_logfile_path(self) -> str | None:
        relaxation_cfg = getattr(self.config, "relaxation", {}) or {}
        value = relaxation_cfg.get("log_file")
        if value in (None, "", False):
            return None
        output_dir = Path(str(self.config.output_dir))
        path = Path(str(value))
        if path.parent == Path("."):
            path = output_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _progress_log_path(self) -> Path | None:
        relaxation_cfg = getattr(self.config, "relaxation", {}) or {}
        value = relaxation_cfg.get("progress_log")
        if value in (None, "", False):
            return None
        if isinstance(value, str) and value.lower() in {"false", "none", "off"}:
            return None
        output_dir = Path(str(self.config.output_dir))
        path = Path(str(value))
        if path.parent == Path("."):
            path = output_dir / path
        return path

    def _progress(self, message: str) -> None:
        relaxation_cfg = getattr(self.config, "relaxation", {}) or {}
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        if bool(relaxation_cfg.get("progress_stdout", True)):
            print(line, flush=True)
        progress_path = self._progress_log_path()
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _remove_empty_vibration_cache(cache_dir: Path) -> int:
        """Remove empty ASE vibration cache files left by interrupted runs."""
        if not cache_dir.is_dir():
            return 0
        removed = 0
        for path in cache_dir.glob("cache.*.json"):
            try:
                if path.stat().st_size != 0:
                    continue
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        return removed

    @staticmethod
    def _che_label(summary: dict[str, Any]) -> str:
        formula = str(
            summary.get("reference_name") or summary.get("formula") or ""
        ).upper()
        if formula == "H2":
            return "h2"
        if formula == "H2O":
            return "h2o"
        if formula == "O2":
            return "o2"
        raise ValueError(f"Unsupported CHE reference label for {formula!r}.")

    @classmethod
    def _che_snippet(cls, summary: dict[str, Any]) -> dict[str, object]:
        label = cls._che_label(summary)
        thermo_key = f"{label}_thermo"
        energy_key = f"{label}_energy_eV"
        che = {
            energy_key: float(summary["potential_energy_eV"]),
            thermo_key: {
                "enabled": True,
                "temperature_K": float(summary["temperature_K"]),
                "pressure_Pa": float(summary["pressure_Pa"]),
                "geometry": str(summary["geometry"]),
                "symmetrynumber": int(summary["symmetrynumber"]),
                "spin": float(summary["spin"]),
                "vib_energies_eV": [
                    float(value) for value in summary.get("vib_energies_eV", [])
                ],
            },
        }
        if summary.get("correction_eV") not in (None, ""):
            che[f"{label}_correction_eV"] = float(summary["correction_eV"])
        return {"che": che}


def run_reference_thermo_stage(reaction_config: SimpleNamespace) -> dict[str, str]:
    raw_stage = getattr(reaction_config, "reference_thermo", {}) or {}
    stage = default_reference_thermo_stage_config()
    if isinstance(raw_stage, dict):
        stage = _deep_update(stage, raw_stage)
    if not bool(stage.get("enabled", False)):
        return {}

    outputs: dict[str, str] = {}
    output_root = Path(str(reaction_config.output_dir)) / str(
        stage.get("output_dir", "references")
    )
    if Path(str(stage.get("output_dir", ""))).is_absolute():
        output_root = Path(str(stage["output_dir"]))

    molecules = stage.get("molecules") or []
    if isinstance(molecules, str):
        molecules = [molecules]
    for molecule_name in molecules:
        molecule_name = str(molecule_name).strip()
        if not molecule_name:
            continue
        molecule_key = molecule_name.lower()
        molecule_overrides = stage.get(molecule_key, {}) or {}
        if not isinstance(molecule_overrides, dict):
            molecule_overrides = {}

        reference = {
            "molecule": molecule_name,
            "vacuum_A": stage.get("vacuum_A", 8.0),
            "temperature_K": _stage_temperature(stage, reaction_config),
            "pressure_Pa": stage.get("pressure_Pa", 101325.0),
        }
        reference.update(molecule_overrides)

        calculator = {
            "calculator": getattr(reaction_config, "calculator", "lj"),
            "lj_cutoff": getattr(reaction_config, "lj_cutoff", 6.0),
            "model": getattr(reaction_config, "model", None),
            "model_file": getattr(reaction_config, "model_file", None),
            "device": getattr(reaction_config, "device", "cpu"),
            "use_kokkos": getattr(reaction_config, "use_kokkos", True),
        }
        if isinstance(stage.get("calculator"), dict):
            calculator.update(stage["calculator"])

        raw_single = {
            "reference": reference,
            "calculator": calculator,
            "output": {
                "output_dir": str(output_root / molecule_key),
                "skip_existing": bool(stage.get("skip_existing", True)),
            },
            "relaxation": stage.get("relaxation", {}),
            "vibrations": stage.get("vibrations", {}),
        }
        workflow = ReferenceThermoWorkflow(_namespace_from_reference_mapping(raw_single))
        molecule_outputs = workflow.run()
        snippet = (
            yaml.safe_load(Path(molecule_outputs["che_snippet_yaml"]).read_text())
            or {}
        )
        che_updates = snippet.get("che", {})
        if isinstance(che_updates, dict):
            current_che = getattr(reaction_config, "che", {}) or {}
            if (
                bool(stage.get("overwrite_che_references", False))
                or _reference_has_che_override(molecule_overrides)
            ):
                reaction_config.che = _deep_update(current_che, che_updates)
            else:
                reaction_config.che = _fill_missing_che_references(
                    current_che,
                    che_updates,
                )
        for key, value in molecule_outputs.items():
            outputs[f"reference_thermo_{molecule_key}_{key}"] = value
    return outputs


def _reference_energy_from_mapping(mapping: dict[str, object]) -> float | None:
    value = mapping.get("energy_eV")
    if value in (None, ""):
        return None
    energy = float(value)
    if not math.isfinite(energy):
        raise ValueError("reference_thermo.<molecule>.energy_eV must be finite.")
    return energy


def _reference_has_che_override(mapping: dict[str, object]) -> bool:
    if _reference_energy_from_mapping(mapping) is not None:
        return True
    correction = mapping.get("correction_eV")
    if correction in (None, ""):
        return False
    return not math.isclose(float(correction), 0.0, rel_tol=0.0, abs_tol=1e-15)


def _fill_missing_che_references(
    current_che: dict[str, object],
    updates: dict[str, object],
) -> dict[str, object]:
    """Merge generated reference snippets without clobbering manual refs."""

    merged = dict(current_che)
    for energy_key, energy_value in updates.items():
        if not energy_key.endswith("_energy_eV"):
            continue
        if merged.get(energy_key) not in (None, ""):
            continue
        merged[energy_key] = energy_value
        label = energy_key.removesuffix("_energy_eV")
        thermo_key = f"{label}_thermo"
        thermo_update = updates.get(thermo_key)
        if isinstance(thermo_update, dict):
            current_thermo = merged.get(thermo_key, {})
            if not isinstance(current_thermo, dict):
                current_thermo = {}
            merged[thermo_key] = _deep_update(current_thermo, thermo_update)
        correction_key = f"{label}_correction_eV"
        if correction_key in updates and merged.get(correction_key) in (None, ""):
            merged[correction_key] = updates[correction_key]
    return merged


def _stage_temperature(stage: dict[str, object], reaction_config: SimpleNamespace) -> float:
    value = stage.get("temperature_K")
    if value not in (None, ""):
        return float(value)
    che = getattr(reaction_config, "che", {}) or {}
    if (
        isinstance(che, dict)
        and che.get("boltzmann_temperature_K") not in (None, "")
    ):
        return float(che["boltzmann_temperature_K"])
    return 303.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Relax an isolated reference molecule and compute IdealGasThermo inputs."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Reference thermo YAML config.",
    )
    args = parser.parse_args()

    workflow = ReferenceThermoWorkflow.from_yaml(args.config)
    outputs = workflow.run()
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
