from __future__ import annotations

import csv
import json
import logging
import numpy as np
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from ase import Atoms
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.io import read, write

from .adsorbate_cmc import AdsorbateCMC
from .adsorbate_gcmc import AdsorbateGCMC
from .alloy_cmc import AlloyCMC
from .replica import ReplicaExchange
from .utils import initialize_surface_adsorbates
from .utils import initialize_alloy_sublattice
from .utils import generate_nonuniform_temperature_grid


TaskRunner = Callable[[dict], dict]
StatusFormatter = Callable[[dict], str]
CalculatorFactory = Callable[[SimpleNamespace, dict], object]
AtomsPreparer = Callable[[object, SimpleNamespace, dict], object]
FunctionalElementsFactory = Callable[[object, SimpleNamespace, dict], tuple[str, ...]]

_RESERVED_CONFIG_KEYS = {
    "mu_values",
    "seeds",
    "devices",
    "n_workers",
    "backend",
    "gpu_ids",
    "ray_address",
    "ray_log_to_driver",
    "ray_num_cpus_per_task",
    "ray_num_gpus_per_task",
    "output_dir",
}

_DEFAULT_ADSORBATE_GCMC_SCAN_CONFIG = {
    "snapshot": None,
    "frame": 0,
    "repeat": [1, 1, 1],
    "fix_below_z": None,
    "calculator": "lj",
    "model": None,
    "model_file": None,
    "use_kokkos": True,
    "devices": ["cuda:0"],
    "lj_cutoff": 6.0,
    "adsorbate": None,
    "adsorbate_anchor_index": 0,
    "temperature": 300.0,
    "mu_values": [-3.0, -2.0, -1.0, 0.0, 1.0],
    "seeds": [81, 82],
    "site_type": ["atop", "bridge"],
    "site_elements": [],
    "surface_side": "top",
    "substrate_elements": [],
    "functional_elements": None,
    "move_mode": "hybrid",
    "site_hop_prob": 0.25,
    "displacement_sigma": 0.25,
    "max_displacement_trials": 20,
    "min_clearance": 0.9,
    "site_match_tol": 0.6,
    "surface_layer_tol": 0.5,
    "termination_clearance": 0.8,
    "vertical_offset": 1.8,
    "w_insert": 1.0,
    "w_delete": 1.0,
    "w_canonical": 1.0,
    "max_n_adsorbates": None,
    "relax": False,
    "relax_steps": 10,
    "fmax": 0.05,
    "enable_hybrid_md": False,
    "md_move_prob": 0.0,
    "md_steps": 0,
    "md_timestep_fs": 1.0,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "md_friction": 0.01,
    "md_planar": False,
    "md_planar_axis": 2,
    "md_init_momenta": True,
    "md_remove_drift": True,
    "nsweeps": 200,
    "write_interval": 10,
    "sample_interval": 2,
    "equilibration": 40,
    "backend": "multiprocessing",
    "n_workers": 1,
    "gpu_ids": [],
    "ray_address": None,
    "ray_log_to_driver": False,
    "ray_num_cpus_per_task": 1,
    "ray_num_gpus_per_task": None,
    "worker_log_to_stdout": False,
    "output_dir": "adsorbate_gcmc_scan",
}

_DEFAULT_ADSORBATE_GCMC_CONFIG = {
    "snapshot": None,
    "frame": 0,
    "repeat": [1, 1, 1],
    "fix_below_z": None,
    "calculator": "lj",
    "model": None,
    "model_file": None,
    "use_kokkos": True,
    "device": "cuda",
    "lj_cutoff": 6.0,
    "adsorbate": None,
    "adsorbate_anchor_index": 0,
    "temperature": 300.0,
    "chemical_potential": 0.0,
    "site_type": ["atop", "bridge"],
    "site_elements": [],
    "surface_side": "top",
    "substrate_elements": [],
    "functional_elements": None,
    "move_mode": "hybrid",
    "site_hop_prob": 0.25,
    "displacement_sigma": 0.25,
    "max_displacement_trials": 20,
    "min_clearance": 0.9,
    "site_match_tol": 0.6,
    "surface_layer_tol": 0.5,
    "termination_clearance": 0.8,
    "vertical_offset": 1.8,
    "w_insert": 1.0,
    "w_delete": 1.0,
    "w_canonical": 1.0,
    "max_n_adsorbates": None,
    "relax": False,
    "relax_steps": 10,
    "fmax": 0.05,
    "enable_hybrid_md": False,
    "md_move_prob": 0.0,
    "md_steps": 0,
    "md_timestep_fs": 1.0,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "md_friction": 0.01,
    "md_planar": False,
    "md_planar_axis": 2,
    "md_init_momenta": True,
    "md_remove_drift": True,
    "nsweeps": 200,
    "write_interval": 10,
    "sample_interval": 2,
    "equilibration": 40,
    "seed": 81,
    "output_prefix": "adsorbate_gcmc",
}

_DEFAULT_ADSORBATE_CMC_CONFIG = {
    "snapshot": None,
    "frame": 0,
    "repeat": [1, 1, 1],
    "fix_below_z": None,
    "calculator": "lj",
    "model": None,
    "model_file": None,
    "use_kokkos": True,
    "device": "cuda",
    "lj_cutoff": 6.0,
    "adsorbate": "H",
    "adsorbate_anchor_index": 0,
    "temperature": 300.0,
    "initialization_mode": "clean_surface",
    "coverage": 1.0,
    "n_adsorbates": None,
    "site_type": ["atop"],
    "site_elements": [],
    "surface_side": "top",
    "substrate_elements": [],
    "functional_elements": None,
    "top_layer_element": None,
    "move_mode": "displacement",
    "site_hop_prob": 0.5,
    "reorientation_prob": 0.2,
    "rotation_max_angle_deg": 25.0,
    "displacement_sigma": 0.6,
    "max_displacement_trials": 20,
    "max_reorientation_trials": None,
    "min_clearance": 0.9,
    "site_match_tol": 0.6,
    "surface_layer_tol": 0.5,
    "termination_clearance": 0.8,
    "bridge_cutoff": None,
    "vertical_offset": 1.8,
    "detach_tol": 3.0,
    "z_max_support": 3.5,
    "relax": False,
    "relax_steps": 20,
    "relax_z_only": False,
    "fmax": 0.05,
    "verbose_relax": False,
    "enable_hybrid_md": False,
    "md_move_prob": 0.1,
    "md_steps": 50,
    "md_timestep_fs": 1.0,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "md_friction": 0.01,
    "md_planar": False,
    "md_planar_axis": 2,
    "md_init_momenta": True,
    "md_remove_drift": True,
    "nsweeps": 200,
    "write_interval": 10,
    "sample_interval": 2,
    "equilibration": 40,
    "seed": 81,
    "resume": False,
    "checkpoint_interval": 100,
    "output_prefix": "adsorbate_cmc",
}

_DEFAULT_ALLOY_CMC_CONFIG = {
    "snapshot": None,
    "frame": 0,
    "repeat": [1, 1, 1],
    "supercell_matrix": None,
    "fix_below_z": None,
    "calculator": "lj",
    "model": None,
    "model_file": None,
    "use_kokkos": True,
    "device": "cuda",
    "lj_cutoff": 6.0,
    "site_element": None,
    "composition": None,
    "initialization_seed": 67,
    "temperature": 300.0,
    "swap_elements": [],
    "swap_mode": "hybrid",
    "hybrid_neighbor_prob": 0.5,
    "neighbor_cutoff": 3.5,
    "neighbor_backend": "auto",
    "neighbor_cache": True,
    "relax": False,
    "relax_steps": 10,
    "local_relax": False,
    "relax_radius": 4.0,
    "fmax": 0.05,
    "enable_hybrid_md": False,
    "md_move_prob": 0.1,
    "md_steps": 50,
    "md_timestep_fs": 1.0,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "md_friction": 0.01,
    "md_planar": False,
    "md_planar_axis": 2,
    "md_init_momenta": True,
    "md_remove_drift": True,
    "nsweeps": 200,
    "write_interval": 10,
    "sample_interval": 1,
    "equilibration": 0,
    "seed": 67,
    "resume": False,
    "checkpoint_interval": 100,
    "output_prefix": "alloy_cmc",
}

_DEFAULT_ALLOY_PT_CONFIG = {
    "snapshot": None,
    "frame": 0,
    "repeat": [1, 1, 1],
    "supercell_matrix": None,
    "fix_below_z": None,
    "calculator": "lj",
    "model": None,
    "model_file": None,
    "use_kokkos": True,
    "device": "cuda",
    "default_dtype": None,
    "lj_cutoff": 6.0,
    "site_element": None,
    "composition": None,
    "initialization_seed": 67,
    "swap_elements": [],
    "swap_mode": "hybrid",
    "hybrid_neighbor_prob": 0.5,
    "neighbor_cutoff": 3.5,
    "neighbor_backend": "auto",
    "neighbor_cache": True,
    "relax": False,
    "relax_steps": 10,
    "local_relax": False,
    "relax_radius": 4.0,
    "fmax": 0.05,
    "enable_hybrid_md": False,
    "md_move_prob": 0.1,
    "md_steps": 50,
    "md_timestep_fs": 1.0,
    "md_ensemble": "nve",
    "md_accept_mode": "potential",
    "md_friction": 0.01,
    "md_planar": False,
    "md_planar_axis": 2,
    "md_init_momenta": True,
    "md_remove_drift": True,
    "T_start": 800.0,
    "T_end": 50.0,
    "T_step": 50.0,
    "n_replicas": None,
    "fine_grid_temps": [],
    "fine_grid_weights": [],
    "fine_grid_strength": 4.0,
    "fine_grid_width": None,
    "grid_space": "temperature",
    "swap_stride": 1,
    "swap_interval": 20,
    "report_interval": 5,
    "sampling_interval": 1,
    "local_eq_fraction": 0.2,
    "checkpoint_interval": 10,
    "resume": False,
    "track_composition": [],
    "seed_nonce": 0,
    "n_cycles": 2,
    "equilibration_cycles": 0,
    "backend": "multiprocessing",
    "n_gpus": 1,
    "workers_per_gpu": 1,
    "ray_address": None,
    "ray_log_to_driver": False,
    "ray_num_cpus_per_task": 1,
    "ray_num_gpus_per_task": None,
    "use_placement_group": False,
    "placement_group_strategy": "SPREAD",
    "remove_placement_group_on_stop": True,
    "shutdown_on_stop": False,
    "stats_file": "replica_stats.csv",
    "results_file": "results.csv",
    "checkpoint_file": "pt_state.pkl",
    "output_dir": "alloy_pt",
}


def format_mu_label(mu: float) -> str:
    return f"{mu:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def configure_mc_logger(log_file: Path, stream: bool = False) -> None:
    logger = logging.getLogger("mc")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def format_adsorbate_gcmc_status(result: dict) -> str:
    return (
        f"done mu={result['mu']:+.3f} seed={result['seed']} "
        f"avgN={result['n_adsorbates_avg']:.3f} acc={result['acceptance']:.2f}%"
    )


def load_snapshot_default(path: Path, frame: int):
    suffix = path.suffix.lower()
    index = frame if suffix in {".traj", ".db", ".extxyz", ".xyz"} else 0
    return read(path, index=index)


def _parse_symbols(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(token for token in value.replace(",", " ").split() if token)
    return tuple(value)


def build_adsorbate_template(name_or_path: str | Atoms) -> tuple[Atoms, int]:
    if isinstance(name_or_path, Atoms):
        return name_or_path.copy(), 0

    key = str(name_or_path).upper()
    if key == "H":
        return Atoms("H", positions=[(0.0, 0.0, 0.0)]), 0
    if key == "OH":
        return Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]), 0
    if key == "OOH":
        return (
            Atoms(
                "OOH",
                positions=[
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.46),
                    (0.79, 0.0, 2.01),
                ],
            ),
            0,
        )

    path = Path(str(name_or_path))
    if path.exists():
        return read(path), 0

    return Atoms(str(name_or_path), positions=[(0.0, 0.0, 0.0)]), 0


def _infer_functional_elements_from_config(
    atoms, cfg: SimpleNamespace, _task: dict
) -> tuple[str, ...]:
    explicit = getattr(cfg, "functional_elements", None)
    if explicit:
        return _parse_symbols(explicit)
    substrate_elements = set(_parse_symbols(getattr(cfg, "substrate_elements", ())))
    return tuple(sorted(set(atoms.get_chemical_symbols()) - substrate_elements))


def _prepare_adsorbate_scan_atoms(atoms, cfg: SimpleNamespace, _task: dict):
    repeat = tuple(getattr(cfg, "repeat", (1, 1, 1)))
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)

    fix_below_z = getattr(cfg, "fix_below_z", None)
    if fix_below_z is not None:
        fixed_indices = [atom.index for atom in atoms if atom.position[2] < fix_below_z]
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return atoms


def _prepare_alloy_atoms(atoms, cfg: SimpleNamespace):
    supercell_matrix = getattr(cfg, "supercell_matrix", None)
    if supercell_matrix is not None:
        atoms = make_supercell(atoms, np.asarray(supercell_matrix, dtype=int))

    repeat = tuple(getattr(cfg, "repeat", (1, 1, 1)))
    if repeat != (1, 1, 1):
        atoms = atoms.repeat(repeat)

    fix_below_z = getattr(cfg, "fix_below_z", None)
    if fix_below_z is not None:
        fixed_indices = [atom.index for atom in atoms if atom.position[2] < fix_below_z]
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))
    return atoms


def build_adsorbate_gcmc_calculator(cfg: SimpleNamespace, task: dict):
    calculator = str(getattr(cfg, "calculator", "lj")).lower()
    if calculator == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones(rc=float(cfg.lj_cutoff))

    if calculator == "mace":
        from mace.calculators import MACECalculator

        model_path = getattr(cfg, "model", None) or getattr(cfg, "model_file", None)
        if not model_path:
            raise ValueError("MACE calculator requires 'model' or 'model_file'.")
        device = task.get("device") or "cpu"
        return MACECalculator(model_paths=[str(model_path)], device=device)

    if calculator == "symmetrix":
        from symmetrix import Symmetrix

        model_file = getattr(cfg, "model_file", None) or getattr(cfg, "model", None)
        if not model_file:
            raise ValueError("Symmetrix calculator requires 'model_file' or 'model'.")
        return Symmetrix(
            model_file=str(model_file),
            use_kokkos=bool(getattr(cfg, "use_kokkos", True)),
        )

    raise ValueError(f"Unsupported calculator '{cfg.calculator}'.")


def build_replica_calculator_spec(
    cfg: SimpleNamespace,
) -> tuple[type, dict]:
    calculator = str(getattr(cfg, "calculator", "lj")).lower()

    if calculator == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones, {"rc": float(cfg.lj_cutoff)}

    if calculator == "mace":
        from mace.calculators import MACECalculator

        model_path = getattr(cfg, "model", None) or getattr(cfg, "model_file", None)
        if not model_path:
            raise ValueError("MACE calculator requires 'model' or 'model_file'.")
        calc_kwargs = {
            "model_paths": [str(model_path)],
            "device": getattr(cfg, "device", "cuda"),
        }
        default_dtype = getattr(cfg, "default_dtype", None)
        if default_dtype is not None:
            calc_kwargs["default_dtype"] = str(default_dtype)
        return MACECalculator, calc_kwargs

    if calculator == "symmetrix":
        from symmetrix import Symmetrix

        model_file = getattr(cfg, "model_file", None) or getattr(cfg, "model", None)
        if not model_file:
            raise ValueError("Symmetrix calculator requires 'model_file' or 'model'.")
        return Symmetrix, {
            "model_file": str(model_file),
            "use_kokkos": bool(getattr(cfg, "use_kokkos", True)),
        }

    raise ValueError(f"Unsupported calculator '{cfg.calculator}'.")


def _deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path_fields(flat_config: dict, base_dir: Path) -> dict:
    for key in ("snapshot", "model", "model_file", "output_dir"):
        value = flat_config.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            flat_config[key] = str((base_dir / path).resolve())
        else:
            flat_config[key] = str(path)
    return flat_config


def load_adsorbate_gcmc_scan_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Adsorbate GCMC scan config must be a mapping.")

    if any(
        key in raw
        for key in ("system", "gcmc", "backend", "calculator", "output")
    ):
        merged = _deep_update(
            {
                "system": {},
                "gcmc": {},
                "backend": {},
                "calculator": {},
                "output": {},
            },
            raw,
        )
        flat_config = dict(_DEFAULT_ADSORBATE_GCMC_SCAN_CONFIG)
        for section in ("system", "gcmc", "backend", "calculator", "output"):
            flat_config.update(merged.get(section, {}))
    else:
        flat_config = dict(_DEFAULT_ADSORBATE_GCMC_SCAN_CONFIG)
        flat_config.update(raw)

    if "interval" in flat_config:
        flat_config["write_interval"] = flat_config.pop("interval")
    flat_config = _resolve_path_fields(flat_config, config_path.parent)
    adsorbate_value = flat_config.get("adsorbate")
    if isinstance(adsorbate_value, str):
        adsorbate_path = config_path.parent / adsorbate_value
        if (
            Path(adsorbate_value).is_absolute()
            or adsorbate_path.exists()
            or Path(adsorbate_value).suffix
        ):
            path = Path(adsorbate_value)
            if not path.is_absolute():
                flat_config["adsorbate"] = str(adsorbate_path.resolve())
            else:
                flat_config["adsorbate"] = str(path)
    return SimpleNamespace(**flat_config)


def load_adsorbate_gcmc_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Adsorbate GCMC config must be a mapping.")

    if any(key in raw for key in ("system", "gcmc", "calculator", "output")):
        merged = _deep_update(
            {
                "system": {},
                "gcmc": {},
                "calculator": {},
                "output": {},
            },
            raw,
        )
        flat_config = dict(_DEFAULT_ADSORBATE_GCMC_CONFIG)
        for section in ("system", "gcmc", "calculator", "output"):
            flat_config.update(merged.get(section, {}))
    else:
        flat_config = dict(_DEFAULT_ADSORBATE_GCMC_CONFIG)
        flat_config.update(raw)

    if "interval" in flat_config:
        flat_config["write_interval"] = flat_config.pop("interval")
    flat_config = _resolve_path_fields(flat_config, config_path.parent)
    adsorbate_value = flat_config.get("adsorbate")
    if isinstance(adsorbate_value, str):
        adsorbate_path = config_path.parent / adsorbate_value
        if (
            Path(adsorbate_value).is_absolute()
            or adsorbate_path.exists()
            or Path(adsorbate_value).suffix
        ):
            path = Path(adsorbate_value)
            if not path.is_absolute():
                flat_config["adsorbate"] = str(adsorbate_path.resolve())
            else:
                flat_config["adsorbate"] = str(path)
    output_prefix = flat_config.get("output_prefix")
    if output_prefix is not None:
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.is_absolute():
            flat_config["output_prefix"] = str(
                (config_path.parent / output_prefix_path).resolve()
            )
        else:
            flat_config["output_prefix"] = str(output_prefix_path)
    return SimpleNamespace(**flat_config)


def load_adsorbate_cmc_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Adsorbate CMC config must be a mapping.")

    if any(
        key in raw for key in ("system", "cmc", "calculator", "output")
    ):
        merged = _deep_update(
            {
                "system": {},
                "cmc": {},
                "calculator": {},
                "output": {},
            },
            raw,
        )
        flat_config = dict(_DEFAULT_ADSORBATE_CMC_CONFIG)
        for section in ("system", "cmc", "calculator", "output"):
            flat_config.update(merged.get(section, {}))
    else:
        flat_config = dict(_DEFAULT_ADSORBATE_CMC_CONFIG)
        flat_config.update(raw)

    if "interval" in flat_config:
        flat_config["write_interval"] = flat_config.pop("interval")
    flat_config = _resolve_path_fields(flat_config, config_path.parent)
    adsorbate_value = flat_config.get("adsorbate")
    if isinstance(adsorbate_value, str):
        adsorbate_path = config_path.parent / adsorbate_value
        if (
            Path(adsorbate_value).is_absolute()
            or adsorbate_path.exists()
            or Path(adsorbate_value).suffix
        ):
            path = Path(adsorbate_value)
            if not path.is_absolute():
                flat_config["adsorbate"] = str(adsorbate_path.resolve())
            else:
                flat_config["adsorbate"] = str(path)
    output_prefix = flat_config.get("output_prefix")
    if output_prefix is not None:
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.is_absolute():
            flat_config["output_prefix"] = str(
                (config_path.parent / output_prefix_path).resolve()
            )
        else:
            flat_config["output_prefix"] = str(output_prefix_path)
    return SimpleNamespace(**flat_config)


def load_alloy_cmc_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Alloy CMC config must be a mapping.")

    if any(key in raw for key in ("system", "cmc", "calculator", "output")):
        merged = _deep_update(
            {
                "system": {},
                "cmc": {},
                "calculator": {},
                "output": {},
            },
            raw,
        )
        flat_config = dict(_DEFAULT_ALLOY_CMC_CONFIG)
        for section in ("system", "cmc", "calculator", "output"):
            flat_config.update(merged.get(section, {}))
    else:
        flat_config = dict(_DEFAULT_ALLOY_CMC_CONFIG)
        flat_config.update(raw)

    if "interval" in flat_config:
        flat_config["write_interval"] = flat_config.pop("interval")
    flat_config = _resolve_path_fields(flat_config, config_path.parent)
    output_prefix = flat_config.get("output_prefix")
    if output_prefix is not None:
        output_prefix_path = Path(output_prefix)
        if not output_prefix_path.is_absolute():
            flat_config["output_prefix"] = str(
                (config_path.parent / output_prefix_path).resolve()
            )
        else:
            flat_config["output_prefix"] = str(output_prefix_path)
    return SimpleNamespace(**flat_config)


def load_alloy_pt_config(config_path: str | Path) -> SimpleNamespace:
    import yaml

    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Alloy PT config must be a mapping.")

    if any(
        key in raw
        for key in ("system", "pt", "mc", "calculator", "backend", "output")
    ):
        merged = _deep_update(
            {
                "system": {},
                "pt": {},
                "mc": {},
                "calculator": {},
                "backend": {},
                "output": {},
            },
            raw,
        )
        flat_config = dict(_DEFAULT_ALLOY_PT_CONFIG)
        for section in ("system", "pt", "mc", "calculator", "backend", "output"):
            flat_config.update(merged.get(section, {}))
    else:
        flat_config = dict(_DEFAULT_ALLOY_PT_CONFIG)
        flat_config.update(raw)

    flat_config = _resolve_path_fields(flat_config, config_path.parent)
    output_dir = Path(flat_config["output_dir"])
    for key in ("stats_file", "results_file", "checkpoint_file", "output_dir"):
        value = flat_config.get(key)
        if value is None:
            continue
        path = Path(value)
        if key == "output_dir":
            continue
        if not path.is_absolute():
            flat_config[key] = str((output_dir / path).resolve())
        else:
            flat_config[key] = str(path)
    return SimpleNamespace(**flat_config)


def run_tasks_with_multiprocessing(
    tasks: list[dict],
    run_one: TaskRunner,
    *,
    n_workers: int,
    status_formatter: StatusFormatter | None = None,
) -> list[dict]:
    results = []
    if n_workers == 1:
        for task in tasks:
            result = run_one(task)
            results.append(result)
            if status_formatter is not None:
                print(status_formatter(result))
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            if status_formatter is not None:
                print(status_formatter(result))
    return results


def _ray_run_one(task: dict, run_one: TaskRunner) -> dict:
    return run_one(task)


def run_tasks_with_ray(
    tasks: list[dict],
    run_one: TaskRunner,
    *,
    address: str | None = None,
    log_to_driver: bool = False,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
    status_formatter: StatusFormatter | None = None,
) -> list[dict]:
    import ray

    init_kwargs = {"log_to_driver": bool(log_to_driver)}
    if address:
        init_kwargs["address"] = address

    ray.init(**init_kwargs)
    remote_run_one = ray.remote(
        num_cpus=float(num_cpus), num_gpus=float(num_gpus)
    )(_ray_run_one)

    results = []
    pending = [remote_run_one.remote(task, run_one) for task in tasks]
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        result = ray.get(ready[0])
        results.append(result)
        if status_formatter is not None:
            print(status_formatter(result))

    ray.shutdown()
    return results


def write_adsorbate_gcmc_scan_summary(results: list[dict], summary_file: Path) -> None:
    with open(summary_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mu_eV",
                "seed",
                "device",
                "T_K",
                "energy_eV",
                "cv_eV_per_K",
                "acceptance_pct",
                "n_adsorbates_final",
                "n_adsorbates_avg",
                "insert_attempted",
                "insert_accepted",
                "delete_attempted",
                "delete_accepted",
                "canonical_attempted",
                "canonical_accepted",
                "n_hist_json",
                "run_dir",
                "traj_file",
                "thermo_file",
                "checkpoint_file",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    f"{row['mu']:.8f}",
                    row["seed"],
                    row["device"],
                    f"{row['T']:.6f}",
                    f"{row['energy']:.10f}",
                    f"{row['cv']:.10f}",
                    f"{row['acceptance']:.6f}",
                    row["n_adsorbates"],
                    f"{row['n_adsorbates_avg']:.10f}",
                    row["insert_attempted"],
                    row["insert_accepted"],
                    row["delete_attempted"],
                    row["delete_accepted"],
                    row["canonical_attempted"],
                    row["canonical_accepted"],
                    json.dumps(row["n_hist"], sort_keys=True),
                    row["run_dir"],
                    row["traj_file"],
                    row["thermo_file"],
                    row["checkpoint_file"],
                ]
            )


def format_adsorbate_cmc_status(stats: dict) -> str:
    return (
        f"acc={stats['acceptance']:.2f}% "
        f"E={stats['energy']:.6f} eV "
        f"Cv={stats['cv']:.6f} eV/K"
    )


def format_adsorbate_gcmc_run_status(stats: dict) -> str:
    return (
        f"acc={stats['acceptance']:.2f}% "
        f"E={stats['energy']:.6f} eV "
        f"Navg={stats['n_adsorbates_avg']:.3f}"
    )


def format_alloy_cmc_status(stats: dict) -> str:
    return (
        f"acc={stats['acceptance']:.2f}% "
        f"E={stats['energy']:.6f} eV "
        f"Cv={stats['cv']:.6f} eV/K"
    )


def format_alloy_pt_status(temps: list[float]) -> str:
    return "PT temperatures [K]: " + np.array2string(np.asarray(temps), precision=1)


class AdsorbateGCMCScanWorkflow:
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AdsorbateGCMCScanWorkflow":
        config = load_adsorbate_gcmc_scan_config(config_path)
        return cls(
            config,
            calculator_factory=build_adsorbate_gcmc_calculator,
            atoms_preparer=_prepare_adsorbate_scan_atoms,
            functional_elements_factory=_infer_functional_elements_from_config,
        )

    def __init__(
        self,
        config,
        *,
        calculator_factory: CalculatorFactory,
        atoms_preparer: AtomsPreparer | None = None,
        functional_elements_factory: FunctionalElementsFactory | None = None,
        snapshot_loader=load_snapshot_default,
        status_formatter: StatusFormatter = format_adsorbate_gcmc_status,
        summary_writer=write_adsorbate_gcmc_scan_summary,
        extra_reserved_keys: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.calculator_factory = calculator_factory
        self.atoms_preparer = atoms_preparer
        self.functional_elements_factory = functional_elements_factory
        self.snapshot_loader = snapshot_loader
        self.status_formatter = status_formatter
        self.summary_writer = summary_writer
        self.extra_reserved_keys = set(extra_reserved_keys)

    def _build_adsorbate_template(self) -> tuple[Atoms, int]:
        adsorbate_value = getattr(self.config, "adsorbate", None)
        if adsorbate_value is None:
            adsorbate_value = getattr(self.config, "adsorbate_element", "H")
        template, default_anchor_index = build_adsorbate_template(adsorbate_value)
        anchor_index = int(
            getattr(self.config, "adsorbate_anchor_index", default_anchor_index)
        )
        return template, anchor_index

    def _ray_num_gpus_per_task(self) -> float:
        value = getattr(self.config, "ray_num_gpus_per_task", None)
        if value is not None:
            return float(value)
        return 1.0 if getattr(self.config, "calculator", None) == "mace" else 0.0

    def _base_config(self) -> dict:
        reserved = _RESERVED_CONFIG_KEYS | self.extra_reserved_keys
        return {
            key: value
            for key, value in self.config.__dict__.items()
            if key not in reserved
        }

    def _build_tasks(self, out_dir: Path, backend: str) -> list[dict]:
        tasks = []
        devices = tuple(getattr(self.config, "devices", ()))
        gpu_ids = tuple(getattr(self.config, "gpu_ids", ()))
        base_config = self._base_config()
        task_id = 0

        for mu in self.config.mu_values:
            mu_dir = out_dir / f"mu_{format_mu_label(float(mu))}"
            for seed in self.config.seeds:
                if backend == "multiprocessing":
                    device = devices[task_id % len(devices)] if devices else None
                    gpu_id = gpu_ids[task_id % len(gpu_ids)] if gpu_ids else None
                else:
                    device = "cuda" if self._ray_num_gpus_per_task() > 0.0 else "cpu"
                    gpu_id = None

                tasks.append(
                    {
                        "mu": float(mu),
                        "seed": int(seed),
                        "device": device,
                        "gpu_id": gpu_id,
                        "stem": (
                            f"mu_{format_mu_label(float(mu))}_seed_{int(seed):03d}"
                        ),
                        "run_dir": str(mu_dir),
                        "config": base_config,
                    }
                )
                task_id += 1

        return tasks

    def _run_one(self, task: dict) -> dict:
        gpu_id = task.get("gpu_id")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        run_dir = Path(task["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = SimpleNamespace(**task["config"])
        stem = task["stem"]
        configure_mc_logger(
            run_dir / f"{stem}.log",
            stream=bool(getattr(cfg, "worker_log_to_stdout", False)),
        )

        atoms = self.snapshot_loader(Path(cfg.snapshot), int(cfg.frame))
        if self.atoms_preparer is not None:
            prepared = self.atoms_preparer(atoms, cfg, task)
            if prepared is not None:
                atoms = prepared

        calculator = self.calculator_factory(cfg, task)
        substrate_elements = _parse_symbols(getattr(cfg, "substrate_elements", ()))
        if self.functional_elements_factory is not None:
            functional_elements = tuple(
                self.functional_elements_factory(atoms, cfg, task)
            )
        else:
            functional_elements = _parse_symbols(
                getattr(cfg, "functional_elements", ())
            )

        adsorbate_template, anchor_index = self._build_adsorbate_template()

        sim = AdsorbateGCMC(
            atoms=atoms,
            calculator=calculator,
            mu=float(task["mu"]),
            T=cfg.temperature,
            nsteps=cfg.nsweeps,
            adsorbate_element=adsorbate_template[anchor_index].symbol,
            adsorbate=adsorbate_template,
            adsorbate_anchor_index=anchor_index,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            site_elements=_parse_symbols(getattr(cfg, "site_elements", ())),
            surface_side=getattr(cfg, "surface_side", "top"),
            site_type=cfg.site_type,
            move_mode=cfg.move_mode,
            site_hop_prob=cfg.site_hop_prob,
            displacement_sigma=cfg.displacement_sigma,
            max_displacement_trials=cfg.max_displacement_trials,
            min_clearance=cfg.min_clearance,
            site_match_tol=cfg.site_match_tol,
            surface_layer_tol=cfg.surface_layer_tol,
            termination_clearance=cfg.termination_clearance,
            vertical_offset=cfg.vertical_offset,
            w_insert=cfg.w_insert,
            w_delete=cfg.w_delete,
            w_canonical=cfg.w_canonical,
            max_n_adsorbates=cfg.max_n_adsorbates,
            relax=cfg.relax,
            relax_steps=cfg.relax_steps,
            fmax=cfg.fmax,
            enable_hybrid_md=cfg.enable_hybrid_md,
            md_move_prob=getattr(cfg, "md_move_prob", 0.0),
            md_steps=getattr(cfg, "md_steps", 0),
            md_timestep_fs=getattr(cfg, "md_timestep_fs", 1.0),
            md_ensemble=getattr(cfg, "md_ensemble", "nve"),
            md_accept_mode=getattr(cfg, "md_accept_mode", "potential"),
            md_friction=getattr(cfg, "md_friction", 0.01),
            md_planar=getattr(cfg, "md_planar", False),
            md_planar_axis=getattr(cfg, "md_planar_axis", 2),
            md_init_momenta=getattr(cfg, "md_init_momenta", True),
            md_remove_drift=getattr(cfg, "md_remove_drift", True),
            traj_file=str(run_dir / f"{stem}.traj"),
            thermo_file=str(run_dir / f"{stem}.dat"),
            checkpoint_file=str(run_dir / f"{stem}.pkl"),
            checkpoint_interval=0,
            seed=int(task["seed"]),
            allow_ambiguous_empty_adsorbates=True,
        )

        stats = sim.run(
            nsweeps=cfg.nsweeps,
            traj_file=str(run_dir / f"{stem}.traj"),
            interval=cfg.write_interval,
            sample_interval=cfg.sample_interval,
            equilibration=cfg.equilibration,
        )
        return {
            "mu": float(task["mu"]),
            "seed": int(task["seed"]),
            "device": (
                task["device"]
                if task.get("device") is not None
                else (task["gpu_id"] if task.get("gpu_id") is not None else "cpu")
            ),
            "run_dir": str(run_dir),
            "traj_file": str(run_dir / f"{stem}.traj"),
            "thermo_file": str(run_dir / f"{stem}.dat"),
            "checkpoint_file": str(run_dir / f"{stem}.pkl"),
            **stats,
        }

    def run(self) -> list[dict]:
        out_dir = Path(self.config.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        backend = str(self.config.backend).lower()
        if backend not in {"multiprocessing", "ray"}:
            raise ValueError("CONFIG.backend must be 'multiprocessing' or 'ray'.")
        if backend == "multiprocessing" and int(self.config.n_workers) < 1:
            raise ValueError("CONFIG.n_workers must be >= 1.")

        tasks = self._build_tasks(out_dir, backend)

        print(f"Launching {len(tasks)} GCMC jobs at T={self.config.temperature:.1f} K")
        print(f"mu values: {tuple(float(mu) for mu in self.config.mu_values)}")
        print(f"seeds: {tuple(int(seed) for seed in self.config.seeds)}")
        print(f"backend: {backend}")
        if backend == "multiprocessing":
            if getattr(self.config, "devices", None):
                print(f"devices: {tuple(self.config.devices)}")
            else:
                print(
                    f"gpu ids: {tuple(getattr(self.config, 'gpu_ids', (None,)))}"
                )
            print(f"workers: {self.config.n_workers}")
        else:
            print(
                "ray address: "
                f"{self.config.ray_address or os.getenv('RAY_ADDRESS', 'local')}"
            )
            print(f"ray task cpus: {self.config.ray_num_cpus_per_task}")
            print(f"ray task gpus: {self._ray_num_gpus_per_task()}")
        print(f"output dir: {out_dir}")

        if backend == "ray":
            results = run_tasks_with_ray(
                tasks,
                self._run_one,
                address=self.config.ray_address or os.getenv("RAY_ADDRESS"),
                log_to_driver=bool(self.config.ray_log_to_driver),
                num_cpus=float(self.config.ray_num_cpus_per_task),
                num_gpus=self._ray_num_gpus_per_task(),
                status_formatter=self.status_formatter,
            )
        else:
            results = run_tasks_with_multiprocessing(
                tasks,
                self._run_one,
                n_workers=int(self.config.n_workers),
                status_formatter=self.status_formatter,
            )

        results.sort(key=lambda row: (row["mu"], row["seed"]))
        summary_file = out_dir / "summary.csv"
        self.summary_writer(results, summary_file)
        print(f"Wrote summary: {summary_file}")
        return results


class AdsorbateCMCWorkflow:
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AdsorbateCMCWorkflow":
        config = load_adsorbate_cmc_config(config_path)
        return cls(config, calculator_factory=build_adsorbate_gcmc_calculator)

    def __init__(
        self,
        config,
        *,
        calculator_factory: CalculatorFactory,
        snapshot_loader=load_snapshot_default,
        status_formatter: Callable[[dict], str] = format_adsorbate_cmc_status,
    ) -> None:
        self.config = config
        self.calculator_factory = calculator_factory
        self.snapshot_loader = snapshot_loader
        self.status_formatter = status_formatter

    def _load_atoms(self):
        atoms = self.snapshot_loader(Path(self.config.snapshot), int(self.config.frame))
        return _prepare_adsorbate_scan_atoms(atoms, self.config, {})

    def _build_output_paths(self) -> dict[str, str]:
        prefix = Path(self.config.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return {
            "traj_file": str(prefix.with_suffix(".traj")),
            "thermo_file": str(prefix.with_suffix(".dat")),
            "checkpoint_file": str(prefix.with_suffix(".pkl")),
            "initial_traj_file": str(prefix.parent / f"{prefix.name}_initial.traj"),
        }

    def _build_adsorbate_template(self) -> tuple[Atoms, int]:
        template, default_anchor_index = build_adsorbate_template(self.config.adsorbate)
        anchor_index = int(
            getattr(self.config, "adsorbate_anchor_index", default_anchor_index)
        )
        return template, anchor_index

    def _build_simulation(self):
        cfg = self.config
        atoms = self._load_atoms()
        calculator = self.calculator_factory(cfg, {"device": getattr(cfg, "device", None)})
        output_paths = self._build_output_paths()
        substrate_elements = _parse_symbols(getattr(cfg, "substrate_elements", ()))
        functional_elements = _infer_functional_elements_from_config(atoms, cfg, {})
        adsorbate_template, anchor_index = self._build_adsorbate_template()
        anchor_symbol = adsorbate_template[anchor_index].symbol
        init_mode = str(getattr(cfg, "initialization_mode", "clean_surface")).lower()

        common_kwargs = dict(
            T=cfg.temperature,
            adsorbate_element=anchor_symbol,
            adsorbate=adsorbate_template,
            adsorbate_anchor_index=anchor_index,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            top_layer_element=getattr(cfg, "top_layer_element", None),
            site_elements=_parse_symbols(getattr(cfg, "site_elements", ())),
            surface_side=getattr(cfg, "surface_side", "top"),
            site_type=cfg.site_type,
            move_mode=cfg.move_mode,
            site_hop_prob=cfg.site_hop_prob,
            reorientation_prob=getattr(cfg, "reorientation_prob", 0.2),
            rotation_max_angle_deg=getattr(cfg, "rotation_max_angle_deg", 25.0),
            displacement_sigma=cfg.displacement_sigma,
            max_displacement_trials=cfg.max_displacement_trials,
            max_reorientation_trials=getattr(cfg, "max_reorientation_trials", None),
            min_clearance=cfg.min_clearance,
            site_match_tol=cfg.site_match_tol,
            surface_layer_tol=cfg.surface_layer_tol,
            termination_clearance=cfg.termination_clearance,
            bridge_cutoff=getattr(cfg, "bridge_cutoff", None),
            z_max_support=getattr(cfg, "z_max_support", 3.5),
            vertical_offset=cfg.vertical_offset,
            detach_tol=getattr(cfg, "detach_tol", 3.0),
            relax=cfg.relax,
            relax_steps=cfg.relax_steps,
            relax_z_only=getattr(cfg, "relax_z_only", False),
            fmax=cfg.fmax,
            verbose_relax=getattr(cfg, "verbose_relax", False),
            traj_file=output_paths["traj_file"],
            thermo_file=output_paths["thermo_file"],
            checkpoint_file=output_paths["checkpoint_file"],
            checkpoint_interval=int(getattr(cfg, "checkpoint_interval", 100)),
            seed=int(cfg.seed),
            resume=bool(getattr(cfg, "resume", False)),
            enable_hybrid_md=bool(getattr(cfg, "enable_hybrid_md", False)),
            md_move_prob=float(getattr(cfg, "md_move_prob", 0.1)),
            md_steps=int(getattr(cfg, "md_steps", 50)),
            md_timestep_fs=float(getattr(cfg, "md_timestep_fs", 1.0)),
            md_ensemble=getattr(cfg, "md_ensemble", "nve"),
            md_accept_mode=getattr(cfg, "md_accept_mode", "potential"),
            md_friction=float(getattr(cfg, "md_friction", 0.01)),
            md_planar=bool(getattr(cfg, "md_planar", False)),
            md_planar_axis=int(getattr(cfg, "md_planar_axis", 2)),
            md_init_momenta=bool(getattr(cfg, "md_init_momenta", True)),
            md_remove_drift=bool(getattr(cfg, "md_remove_drift", True)),
        )

        init_summary = {
            "initialization_mode": init_mode,
            "snapshot": str(cfg.snapshot),
            "frame": int(cfg.frame),
        }

        if init_mode == "clean_surface":
            cmc = AdsorbateCMC.from_clean_surface(
                atoms=atoms,
                calculator=calculator,
                coverage=float(cfg.coverage),
                initial_traj_file=output_paths["initial_traj_file"],
                **common_kwargs,
            )
            init_summary["coverage"] = float(cfg.coverage)
        elif init_mode == "fixed_count":
            n_adsorbates = getattr(cfg, "n_adsorbates", None)
            if n_adsorbates is None:
                raise ValueError(
                    "initialization_mode='fixed_count' requires n_adsorbates."
                )
            atoms_with_ads, support_indices, candidate_support_indices = (
                initialize_surface_adsorbates(
                    atoms,
                    adsorbate=adsorbate_template,
                    n_adsorbates=int(n_adsorbates),
                    site_elements=_parse_symbols(getattr(cfg, "site_elements", ())),
                    surface_side=getattr(cfg, "surface_side", "top"),
                    site_types=cfg.site_type,
                    layer_tol=cfg.surface_layer_tol,
                    xy_tol=cfg.site_match_tol,
                    bridge_cutoff=getattr(cfg, "bridge_cutoff", None),
                    support_xy_tol=max(1.2, 2.5 * float(cfg.site_match_tol)),
                    vertical_offset=cfg.vertical_offset,
                    termination_elements=functional_elements,
                    min_termination_dist=cfg.termination_clearance,
                    anchor_index=anchor_index,
                    seed=int(cfg.seed),
                )
            )
            write(output_paths["initial_traj_file"], atoms_with_ads)
            cmc = AdsorbateCMC(
                atoms=atoms_with_ads,
                calculator=calculator,
                **common_kwargs,
            )
            init_summary.update(
                {
                    "n_adsorbates": int(n_adsorbates),
                    "candidate_support_count": int(len(candidate_support_indices)),
                    "selected_support_indices": np.asarray(support_indices, dtype=int).tolist()
                    if "np" in globals()
                    else list(support_indices),
                }
            )
        elif init_mode == "preloaded":
            cmc = AdsorbateCMC(
                atoms=atoms,
                calculator=calculator,
                **common_kwargs,
            )
        else:
            raise ValueError(
                "initialization_mode must be 'clean_surface', 'fixed_count', or 'preloaded'."
            )

        return cmc, output_paths, init_summary

    def run(self) -> dict:
        cmc, output_paths, init_summary = self._build_simulation()
        print(f"Loaded snapshot: {init_summary['snapshot']}")
        print(f"Frame: {init_summary['frame']}")
        print(f"Initialization mode: {init_summary['initialization_mode']}")
        if "coverage" in init_summary:
            print(f"Coverage: {init_summary['coverage']}")
        if "n_adsorbates" in init_summary:
            print(f"Fixed loading: {init_summary['n_adsorbates']}")
            print(
                "Candidate top-surface support atoms: "
                f"{init_summary['candidate_support_count']}"
            )
            print(
                "Initial support indices: "
                f"{tuple(init_summary['selected_support_indices'])}"
            )
        print(f"Move mode: {self.config.move_mode}")
        print(
            "Hybrid MD:",
            bool(getattr(self.config, "enable_hybrid_md", False)),
            (
                f"(prob={getattr(self.config, 'md_move_prob', 0.0)}, "
                f"steps={getattr(self.config, 'md_steps', 0)}, "
                f"dt_fs={getattr(self.config, 'md_timestep_fs', 1.0)}, "
                f"planar={getattr(self.config, 'md_planar', False)})"
            ),
        )

        stats = cmc.run(
            nsweeps=int(self.config.nsweeps),
            traj_file=output_paths["traj_file"],
            interval=int(self.config.write_interval),
            sample_interval=int(self.config.sample_interval),
            equilibration=int(self.config.equilibration),
        )
        print(f"Final AdsorbateCMC stats: {stats}")
        print(self.status_formatter(stats))
        return stats


class AdsorbateGCMCWorkflow:
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AdsorbateGCMCWorkflow":
        config = load_adsorbate_gcmc_config(config_path)
        return cls(config, calculator_factory=build_adsorbate_gcmc_calculator)

    def __init__(
        self,
        config,
        *,
        calculator_factory: CalculatorFactory,
        snapshot_loader=load_snapshot_default,
        status_formatter: Callable[[dict], str] = format_adsorbate_gcmc_run_status,
    ) -> None:
        self.config = config
        self.calculator_factory = calculator_factory
        self.snapshot_loader = snapshot_loader
        self.status_formatter = status_formatter

    def _load_atoms(self):
        atoms = self.snapshot_loader(Path(self.config.snapshot), int(self.config.frame))
        return _prepare_adsorbate_scan_atoms(atoms, self.config, {})

    def _build_output_paths(self) -> dict[str, str]:
        prefix = Path(self.config.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return {
            "traj_file": str(prefix.with_suffix(".traj")),
            "thermo_file": str(prefix.with_suffix(".dat")),
            "checkpoint_file": str(prefix.with_suffix(".pkl")),
        }

    def _build_adsorbate_template(self) -> tuple[Atoms, int]:
        adsorbate_value = getattr(self.config, "adsorbate", None)
        if adsorbate_value is None:
            adsorbate_value = getattr(self.config, "adsorbate_element", "H")
        template, default_anchor_index = build_adsorbate_template(adsorbate_value)
        anchor_index = int(
            getattr(self.config, "adsorbate_anchor_index", default_anchor_index)
        )
        return template, anchor_index

    def run(self) -> dict:
        cfg = self.config
        atoms = self._load_atoms()
        calculator = self.calculator_factory(cfg, {"device": getattr(cfg, "device", None)})
        output_paths = self._build_output_paths()
        substrate_elements = _parse_symbols(getattr(cfg, "substrate_elements", ()))
        functional_elements = _infer_functional_elements_from_config(atoms, cfg, {})
        adsorbate_template, anchor_index = self._build_adsorbate_template()

        sim = AdsorbateGCMC(
            atoms=atoms,
            calculator=calculator,
            mu=float(cfg.chemical_potential),
            T=cfg.temperature,
            nsteps=cfg.nsweeps,
            adsorbate_element=adsorbate_template[anchor_index].symbol,
            adsorbate=adsorbate_template,
            adsorbate_anchor_index=anchor_index,
            substrate_elements=substrate_elements,
            functional_elements=functional_elements,
            site_elements=_parse_symbols(getattr(cfg, "site_elements", ())),
            surface_side=getattr(cfg, "surface_side", "top"),
            site_type=cfg.site_type,
            move_mode=cfg.move_mode,
            site_hop_prob=cfg.site_hop_prob,
            displacement_sigma=cfg.displacement_sigma,
            max_displacement_trials=cfg.max_displacement_trials,
            min_clearance=cfg.min_clearance,
            site_match_tol=cfg.site_match_tol,
            surface_layer_tol=cfg.surface_layer_tol,
            termination_clearance=cfg.termination_clearance,
            vertical_offset=cfg.vertical_offset,
            w_insert=cfg.w_insert,
            w_delete=cfg.w_delete,
            w_canonical=cfg.w_canonical,
            max_n_adsorbates=cfg.max_n_adsorbates,
            relax=cfg.relax,
            relax_steps=cfg.relax_steps,
            fmax=cfg.fmax,
            enable_hybrid_md=cfg.enable_hybrid_md,
            md_move_prob=getattr(cfg, "md_move_prob", 0.0),
            md_steps=getattr(cfg, "md_steps", 0),
            md_timestep_fs=getattr(cfg, "md_timestep_fs", 1.0),
            md_ensemble=getattr(cfg, "md_ensemble", "nve"),
            md_accept_mode=getattr(cfg, "md_accept_mode", "potential"),
            md_friction=getattr(cfg, "md_friction", 0.01),
            md_planar=getattr(cfg, "md_planar", False),
            md_planar_axis=getattr(cfg, "md_planar_axis", 2),
            md_init_momenta=getattr(cfg, "md_init_momenta", True),
            md_remove_drift=getattr(cfg, "md_remove_drift", True),
            traj_file=output_paths["traj_file"],
            thermo_file=output_paths["thermo_file"],
            checkpoint_file=output_paths["checkpoint_file"],
            checkpoint_interval=0,
            seed=int(cfg.seed),
            allow_ambiguous_empty_adsorbates=True,
        )

        print(f"Loaded snapshot: {cfg.snapshot}")
        print(f"Frame: {cfg.frame}")
        print(f"T = {cfg.temperature:.1f} K")
        print(f"mu = {float(cfg.chemical_potential):+.6f} eV")
        print(f"Move mode: {cfg.move_mode}")
        print(
            "Hybrid MD:",
            bool(getattr(cfg, "enable_hybrid_md", False)),
            (
                f"(prob={getattr(cfg, 'md_move_prob', 0.0)}, "
                f"steps={getattr(cfg, 'md_steps', 0)}, "
                f"dt_fs={getattr(cfg, 'md_timestep_fs', 1.0)}, "
                f"planar={getattr(cfg, 'md_planar', False)})"
            ),
        )

        stats = sim.run(
            nsweeps=int(cfg.nsweeps),
            traj_file=output_paths["traj_file"],
            interval=int(cfg.write_interval),
            sample_interval=int(cfg.sample_interval),
            equilibration=int(cfg.equilibration),
        )
        print(f"Final AdsorbateGCMC stats: {stats}")
        print(self.status_formatter(stats))
        return stats


class AlloyCMCWorkflow:
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AlloyCMCWorkflow":
        config = load_alloy_cmc_config(config_path)
        return cls(config, calculator_factory=build_adsorbate_gcmc_calculator)

    def __init__(
        self,
        config,
        *,
        calculator_factory: CalculatorFactory,
        snapshot_loader=load_snapshot_default,
        status_formatter: Callable[[dict], str] = format_alloy_cmc_status,
    ) -> None:
        self.config = config
        self.calculator_factory = calculator_factory
        self.snapshot_loader = snapshot_loader
        self.status_formatter = status_formatter

    def _load_atoms(self):
        atoms = self.snapshot_loader(Path(self.config.snapshot), int(self.config.frame))
        return _prepare_alloy_atoms(atoms, self.config)

    def _build_output_paths(self) -> dict[str, str]:
        prefix = Path(self.config.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        return {
            "traj_file": str(prefix.with_suffix(".traj")),
            "accepted_traj_file": str(prefix.parent / f"{prefix.name}_accepted.traj"),
            "thermo_file": str(prefix.with_suffix(".dat")),
            "checkpoint_file": str(prefix.with_suffix(".pkl")),
        }

    def _build_simulation(self):
        cfg = self.config
        atoms = self._load_atoms()
        init_summary = {
            "snapshot": str(cfg.snapshot),
            "frame": int(cfg.frame),
            "initialized_alloy": False,
        }

        site_element = getattr(cfg, "site_element", None)
        composition = getattr(cfg, "composition", None)
        if site_element is not None and composition:
            init_seed = int(getattr(cfg, "initialization_seed", cfg.seed))
            atoms = initialize_alloy_sublattice(
                atoms=atoms,
                site_element=str(site_element),
                composition=dict(composition),
                seed=init_seed,
            )
            init_summary.update(
                {
                    "initialized_alloy": True,
                    "site_element": str(site_element),
                    "composition": dict(composition),
                    "initialization_seed": init_seed,
                }
            )

        calculator = self.calculator_factory(cfg, {"device": getattr(cfg, "device", None)})
        output_paths = self._build_output_paths()
        swap_elements = _parse_symbols(getattr(cfg, "swap_elements", ()))

        mc = AlloyCMC(
            atoms=atoms,
            calculator=calculator,
            T=float(cfg.temperature),
            swap_elements=list(swap_elements) if swap_elements else None,
            swap_mode=str(cfg.swap_mode),
            hybrid_neighbor_prob=float(cfg.hybrid_neighbor_prob),
            neighbor_cutoff=float(cfg.neighbor_cutoff),
            neighbor_backend=str(cfg.neighbor_backend),
            neighbor_cache=bool(cfg.neighbor_cache),
            relax=bool(cfg.relax),
            relax_steps=int(cfg.relax_steps),
            local_relax=bool(getattr(cfg, "local_relax", False)),
            relax_radius=float(getattr(cfg, "relax_radius", 4.0)),
            fmax=float(cfg.fmax),
            traj_file=output_paths["traj_file"],
            accepted_traj_file=output_paths["accepted_traj_file"],
            thermo_file=output_paths["thermo_file"],
            checkpoint_file=output_paths["checkpoint_file"],
            checkpoint_interval=int(getattr(cfg, "checkpoint_interval", 100)),
            seed=int(cfg.seed),
            resume=bool(getattr(cfg, "resume", False)),
            enable_hybrid_md=bool(getattr(cfg, "enable_hybrid_md", False)),
            md_move_prob=float(getattr(cfg, "md_move_prob", 0.1)),
            md_steps=int(getattr(cfg, "md_steps", 50)),
            md_timestep_fs=float(getattr(cfg, "md_timestep_fs", 1.0)),
            md_ensemble=str(getattr(cfg, "md_ensemble", "nve")),
            md_accept_mode=str(getattr(cfg, "md_accept_mode", "potential")),
            md_friction=float(getattr(cfg, "md_friction", 0.01)),
            md_planar=bool(getattr(cfg, "md_planar", False)),
            md_planar_axis=int(getattr(cfg, "md_planar_axis", 2)),
            md_init_momenta=bool(getattr(cfg, "md_init_momenta", True)),
            md_remove_drift=bool(getattr(cfg, "md_remove_drift", True)),
        )
        return mc, output_paths, init_summary

    def run(self) -> dict:
        mc, output_paths, init_summary = self._build_simulation()
        print(f"Loaded snapshot: {init_summary['snapshot']}")
        print(f"Frame: {init_summary['frame']}")
        if init_summary["initialized_alloy"]:
            print(
                "Initialized alloy sublattice: "
                f"{init_summary['site_element']} -> {init_summary['composition']}"
            )
            print(f"Initialization seed: {init_summary['initialization_seed']}")
        print(f"T = {float(self.config.temperature):.1f} K")
        print(f"Swap mode: {self.config.swap_mode}")
        print(
            "Hybrid MD:",
            bool(getattr(self.config, "enable_hybrid_md", False)),
            (
                f"(prob={getattr(self.config, 'md_move_prob', 0.0)}, "
                f"steps={getattr(self.config, 'md_steps', 0)}, "
                f"dt_fs={getattr(self.config, 'md_timestep_fs', 1.0)}, "
                f"planar={getattr(self.config, 'md_planar', False)})"
            ),
        )

        stats = mc.run(
            nsweeps=int(self.config.nsweeps),
            traj_file=output_paths["traj_file"],
            interval=int(self.config.write_interval),
            sample_interval=int(self.config.sample_interval),
            equilibration=int(self.config.equilibration),
        )
        print(f"Final AlloyCMC stats: {stats}")
        print(self.status_formatter(stats))
        return stats


class AlloyReplicaExchangeWorkflow:
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AlloyReplicaExchangeWorkflow":
        config = load_alloy_pt_config(config_path)
        return cls(config)

    def __init__(self, config) -> None:
        self.config = config

    def _load_atoms(self):
        atoms = load_snapshot_default(Path(self.config.snapshot), int(self.config.frame))
        atoms = _prepare_alloy_atoms(atoms, self.config)
        site_element = getattr(self.config, "site_element", None)
        composition = getattr(self.config, "composition", None)
        init_summary = {
            "snapshot": str(self.config.snapshot),
            "frame": int(self.config.frame),
            "initialized_alloy": False,
        }
        if site_element is not None and composition:
            init_seed = int(getattr(self.config, "initialization_seed", 67))
            atoms = initialize_alloy_sublattice(
                atoms=atoms,
                site_element=str(site_element),
                composition=dict(composition),
                seed=init_seed,
            )
            init_summary.update(
                {
                    "initialized_alloy": True,
                    "site_element": str(site_element),
                    "composition": dict(composition),
                    "initialization_seed": init_seed,
                }
            )
        return atoms, init_summary

    def _mc_kwargs(self) -> dict:
        cfg = self.config
        swap_elements = _parse_symbols(getattr(cfg, "swap_elements", ()))
        kwargs = {
            "swap_elements": list(swap_elements) if swap_elements else None,
            "swap_mode": str(cfg.swap_mode),
            "hybrid_neighbor_prob": float(cfg.hybrid_neighbor_prob),
            "neighbor_cutoff": float(cfg.neighbor_cutoff),
            "neighbor_backend": str(cfg.neighbor_backend),
            "neighbor_cache": bool(cfg.neighbor_cache),
            "relax": bool(cfg.relax),
            "relax_steps": int(cfg.relax_steps),
            "local_relax": bool(getattr(cfg, "local_relax", False)),
            "relax_radius": float(getattr(cfg, "relax_radius", 4.0)),
            "fmax": float(cfg.fmax),
            "checkpoint_interval": int(getattr(cfg, "checkpoint_interval", 10)),
            "enable_hybrid_md": bool(getattr(cfg, "enable_hybrid_md", False)),
            "md_move_prob": float(getattr(cfg, "md_move_prob", 0.1)),
            "md_steps": int(getattr(cfg, "md_steps", 50)),
            "md_timestep_fs": float(getattr(cfg, "md_timestep_fs", 1.0)),
            "md_ensemble": str(getattr(cfg, "md_ensemble", "nve")),
            "md_accept_mode": str(getattr(cfg, "md_accept_mode", "potential")),
            "md_friction": float(getattr(cfg, "md_friction", 0.01)),
            "md_planar": bool(getattr(cfg, "md_planar", False)),
            "md_planar_axis": int(getattr(cfg, "md_planar_axis", 2)),
            "md_init_momenta": bool(getattr(cfg, "md_init_momenta", True)),
            "md_remove_drift": bool(getattr(cfg, "md_remove_drift", True)),
        }
        return kwargs

    def _backend_kwargs(self) -> dict | None:
        if str(self.config.backend).lower() != "ray":
            return None

        actor_options = {
            "num_cpus": float(getattr(self.config, "ray_num_cpus_per_task", 1)),
        }
        num_gpus = getattr(self.config, "ray_num_gpus_per_task", None)
        if num_gpus is not None:
            actor_options["num_gpus"] = float(num_gpus)

        return {
            "init_kwargs": {
                "address": getattr(self.config, "ray_address", None) or "auto",
                "log_to_driver": bool(getattr(self.config, "ray_log_to_driver", False)),
            },
            "actor_options": actor_options,
            "use_placement_group": bool(
                getattr(self.config, "use_placement_group", False)
            ),
            "placement_group_strategy": str(
                getattr(self.config, "placement_group_strategy", "SPREAD")
            ),
            "remove_placement_group_on_stop": bool(
                getattr(self.config, "remove_placement_group_on_stop", True)
            ),
            "shutdown_on_stop": bool(
                getattr(self.config, "shutdown_on_stop", False)
            ),
        }

    def _temperatures(self) -> list[float]:
        cfg = self.config
        if getattr(cfg, "n_replicas", None) is not None:
            return generate_nonuniform_temperature_grid(
                T_start=float(cfg.T_start),
                T_end=float(cfg.T_end),
                n_replicas=int(cfg.n_replicas),
                focus_temps=list(getattr(cfg, "fine_grid_temps", []) or []),
                focus_weights=list(getattr(cfg, "fine_grid_weights", []) or []),
                focus_strength=float(getattr(cfg, "fine_grid_strength", 4.0)),
                focus_width=getattr(cfg, "fine_grid_width", None),
                grid_space=str(getattr(cfg, "grid_space", "temperature")),
            )

        T_step = getattr(cfg, "T_step", None)
        if T_step is None or np.isclose(float(T_step), 0.0):
            raise ValueError("T_step must be non-zero when n_replicas is not set.")
        step = abs(float(T_step))
        if float(cfg.T_start) > float(cfg.T_end):
            return np.arange(float(cfg.T_start), float(cfg.T_end) - step / 2.0, -step).tolist()
        return np.arange(float(cfg.T_start), float(cfg.T_end) + step / 2.0, step).tolist()

    def _relocate_replica_outputs(self, pt: ReplicaExchange, out_dir: Path) -> None:
        for state in pt.replica_states:
            state["traj_file"] = str(out_dir / Path(state["traj_file"]).name)
            state["thermo_file"] = str(out_dir / Path(state["thermo_file"]).name)
            state["checkpoint_file"] = str(out_dir / Path(state["checkpoint_file"]).name)

    def run(self) -> ReplicaExchange:
        cfg = self.config
        atoms, init_summary = self._load_atoms()
        calculator_class, calc_kwargs = build_replica_calculator_spec(cfg)
        mc_kwargs = self._mc_kwargs()
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loaded snapshot: {init_summary['snapshot']}")
        print(f"Frame: {init_summary['frame']}")
        if init_summary["initialized_alloy"]:
            print(
                "Initialized alloy sublattice: "
                f"{init_summary['site_element']} -> {init_summary['composition']}"
            )
            print(f"Initialization seed: {init_summary['initialization_seed']}")

        temps = self._temperatures()
        print(format_alloy_pt_status(temps))
        print(f"Backend: {cfg.backend}")
        print(f"GPUs: {int(cfg.n_gpus)} | Workers/GPU: {int(cfg.workers_per_gpu)}")

        pt = ReplicaExchange.from_auto_config(
            atoms_template=atoms,
            T_start=float(cfg.T_start),
            T_end=float(cfg.T_end),
            T_step=None if getattr(cfg, "n_replicas", None) is not None else float(cfg.T_step),
            calculator_class=calculator_class,
            mc_class=AlloyCMC,
            calc_kwargs=calc_kwargs,
            mc_kwargs=mc_kwargs,
            n_gpus=int(cfg.n_gpus),
            workers_per_gpu=int(cfg.workers_per_gpu),
            swap_stride=int(cfg.swap_stride),
            resume=bool(cfg.resume),
            results_file=str(cfg.results_file),
            stats_file=str(cfg.stats_file),
            checkpoint_file=str(cfg.checkpoint_file),
            track_composition=list(getattr(cfg, "track_composition", []) or []),
            seed_nonce=int(getattr(cfg, "seed_nonce", 0)),
            n_replicas=getattr(cfg, "n_replicas", None),
            fine_grid_temps=list(getattr(cfg, "fine_grid_temps", []) or []),
            fine_grid_weights=list(getattr(cfg, "fine_grid_weights", []) or []),
            fine_grid_strength=float(getattr(cfg, "fine_grid_strength", 4.0)),
            fine_grid_width=getattr(cfg, "fine_grid_width", None),
            grid_space=str(getattr(cfg, "grid_space", "temperature")),
            execution_backend=str(cfg.backend),
            backend_kwargs=self._backend_kwargs(),
            swap_interval=int(cfg.swap_interval),
            report_interval=int(cfg.report_interval),
            sampling_interval=int(cfg.sampling_interval),
            local_eq_fraction=float(cfg.local_eq_fraction),
            checkpoint_interval=int(cfg.checkpoint_interval),
        )
        self._relocate_replica_outputs(pt, out_dir)
        pt.run(
            n_cycles=int(cfg.n_cycles),
            equilibration_cycles=int(getattr(cfg, "equilibration_cycles", 0)),
        )
        return pt
