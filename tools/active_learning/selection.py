#!/usr/bin/env python3
"""Select active-learning structures from a trajectory and prepare VASP jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, read
from ase.io.trajectory import Trajectory as AseTrajectory
from scipy.spatial.distance import cdist

from .models import (
    add_model_arguments,
    build_calculator,
    model_labels,
    release_calculator,
    resolve_calculator_backend,
    resolve_models,
)


DEFAULT_SOAP_SPECIES = ("H", "C", "O", "Ti", "Zr", "Mo")


@dataclass
class ReferenceEntry:
    atoms: Atoms
    model_seen: bool = False
    dft_seen: bool = False
    sources: set[str] = field(default_factory=set)


class Generate:
    """Write a static VASP calculation for one selected structure."""

    def __init__(self, structure, output_dir: Path, functional: str):
        if functional == "PBE0-rVV10_8":
            self.settings = {
                "LDAU": False,
                "ISIF": 1,
                "IBRION": -1,
                "NSW": 0,
                "ISPIN": 2,
                "LCHARG": False,
                "LWAVE": False,
                "KPAR": 4,
                "NELM": 300,
                "ENCUT": 500,
                "NELMDL": 6,
                "GGA": "PE",
                "LHFCALC": True,
                "AEXX": 0.08,
                "HFSCREEN": 0,
                "ALGO": "All",
                "IMIX": 1,
                "AMIX": 0.35,
                "ISMEAR": 0,
                "SIGMA": 0.05,
                "EDIFF": 1e-6,
                "PREC": "Accurate",
                "PRECFOCK": "Fast",
                "LASPH": True,
                "LUSE_VDW": True,
                "IVDW_NL": 2,
                "BPARAM": 6.3,
                "CPARAM": 0.0093,
            }
        elif functional == "PBE-D3":
            self.settings = {
                "ISIF": 1,
                "IBRION": -1,
                "NSW": 0,
                "KPAR": 4,
                "IVDW": 12,
                "LREAL": "Auto",
                "LASPH": True,
                "NELM": 200,
                "HFSCREEN": 0,
                "GGA": "PE",
                "ALGO": "Normal",
                "EDIFF": 1e-6,
                "ISMEAR": 0,
                "SIGMA": 0.02,
                "ENCUT": 500,
                "PREC": "Accurate",
                "ISPIN": 2,
                "LWAVE": False,
                "LCHARG": False,
                "NCORE": 4,
            }
        else:
            raise ValueError(f"Unsupported functional: {functional}")

        self.user_potcar_settings = {
            "Ti": "Ti_sv",
            "Zr": "Zr_sv",
            "Nb": "Nb_sv",
            "Mo": "Mo_sv",
            "V": "V_sv",
            "Cu": "Cu",
        }
        self.output_dir = output_dir
        self.structure = structure

    def scf(self, overwrite: bool = False) -> None:
        from pymatgen.io.vasp.sets import MPStaticSet

        if (self.output_dir / "POSCAR").exists() and not overwrite:
            raise FileExistsError(
                f"{self.output_dir}/POSCAR already exists; use --overwrite to replace it"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        scfset = MPStaticSet(
            self.structure,
            user_incar_settings=self.settings,
            user_potcar_functional="PBE_54",
            user_potcar_settings=self.user_potcar_settings,
            reciprocal_density=250,
        )
        scfset.write_input(str(self.output_dir), potcar_spec=False)
        print(f"Wrote VASP input: {self.output_dir}")


def structure_hash(atoms, decimals: int = 6) -> str:
    """Hash species, cell, PBC, and wrapped positions deterministically."""
    cell = np.round(atoms.cell.array, decimals=decimals)
    pbc = atoms.get_pbc().astype(int)
    if atoms.cell.rank == 3:
        positions = np.round(atoms.get_scaled_positions(wrap=True), decimals=decimals)
        position_kind = "scaled"
    else:
        positions = np.round(atoms.get_positions(), decimals=decimals)
        position_kind = "cartesian"
    rows = sorted(
        (int(number), *map(float, position))
        for number, position in zip(atoms.get_atomic_numbers(), positions)
    )
    payload = {
        "cell": cell.tolist(),
        "pbc": pbc.tolist(),
        "positions": rows,
        "position_kind": position_kind,
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def resolve_input_trajectories(values: list[str] | None) -> list[Path]:
    if values:
        paths = list(
            dict.fromkeys(
                Path(value).expanduser().resolve() for value in values
            )
        )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Input trajectory file(s) not found: " + ", ".join(missing)
            )
        return paths

    candidates = [Path("XDATCAR"), Path("md.traj")]
    for candidate in candidates:
        if candidate.is_file():
            return [candidate.resolve()]
    traj_files = sorted(Path(".").glob("*.traj"))
    if len(traj_files) == 1:
        return [traj_files[0].resolve()]
    if traj_files:
        raise FileNotFoundError(
            "Multiple .traj files found; provide --input or name the input md.traj."
        )
    raise FileNotFoundError(
        "Could not find XDATCAR, md.traj, or a single .traj file; provide --input."
    )


def load_trajectory_frames(
    trajectory_path: Path, frame_stride: int
) -> tuple[list, list[int]]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if trajectory_path.name == "XDATCAR":
        from pymatgen.core.trajectory import Trajectory as PymatgenTrajectory
        from pymatgen.io.ase import AseAtomsAdaptor

        trajectory = PymatgenTrajectory.from_file(str(trajectory_path))
        source_indices = list(range(0, len(trajectory), frame_stride))
        frames = [
            AseAtomsAdaptor.get_atoms(trajectory.get_structure(index))
            for index in source_indices
        ]
    else:
        loaded = read(str(trajectory_path), index=":")
        frames_all = loaded if isinstance(loaded, list) else [loaded]
        source_indices = list(range(0, len(frames_all), frame_stride))
        frames = [frames_all[index].copy() for index in source_indices]
    if not frames:
        raise ValueError(f"No frames loaded from {trajectory_path}")
    print(
        f"Loaded {len(frames)} frames from {trajectory_path} "
        f"(stride={frame_stride})"
    )
    return frames, source_indices


def resolve_reference_pools(
    models: list[Path],
    model_seen_values: list[str] | None,
    dft_seen_values: list[str] | None,
    legacy_values: list[str] | None,
    no_reference_pool: bool,
) -> tuple[list[Path], list[Path]]:
    model_pools: list[Path] = []
    dft_pools: list[Path] = []
    if not no_reference_pool:
        pool_roots = {
            model.parent.parent if model.parent.name == "models" else model.parent
            for model in models
        }
        for root in sorted(pool_roots):
            model_pools.extend(sorted(root.glob("*_train.xyz")))
            dft_pools.extend(sorted(root.glob("*_test.xyz")))
    if model_seen_values:
        model_pools.extend(
            Path(value).expanduser().resolve() for value in model_seen_values
        )
    if dft_seen_values:
        dft_pools.extend(
            Path(value).expanduser().resolve() for value in dft_seen_values
        )
    if legacy_values:
        legacy = [Path(value).expanduser().resolve() for value in legacy_values]
        model_pools.extend(legacy)
        dft_pools.extend(legacy)
    model_pools = list(dict.fromkeys(model_pools))
    dft_pools = list(dict.fromkeys(dft_pools + model_pools))
    missing = [
        str(path)
        for path in model_pools + dft_pools
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Reference pool(s) not found: " + ", ".join(missing))
    return model_pools, dft_pools


def register_reference(
    registry: dict[str, ReferenceEntry],
    atoms: Atoms,
    *,
    model_seen: bool,
    dft_seen: bool,
    source: str,
    decimals: int,
) -> None:
    key = structure_hash(atoms, decimals=decimals)
    if key not in registry:
        registry[key] = ReferenceEntry(atoms=atoms.copy())
    entry = registry[key]
    entry.model_seen = entry.model_seen or model_seen
    entry.dft_seen = entry.dft_seen or dft_seen or model_seen
    entry.sources.add(source)


def periodic_soap_compatible(atoms: Atoms) -> bool:
    return bool(atoms.cell.rank == 3 and np.all(atoms.get_pbc()))


def calculation_structure_file(directory: Path) -> Path | None:
    for name in ("CONTCAR", "POSCAR"):
        path = directory / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def discover_existing_round_structures(
    output_root: Path,
) -> list[tuple[Path, bool, bool]]:
    discovered: list[tuple[Path, bool, bool]] = []
    if not output_root.is_dir():
        return discovered

    for round_dir in sorted(output_root.iterdir()):
        if not round_dir.is_dir():
            continue
        if round_dir.name.isdigit():
            calculation_dirs = [
                path
                for path in round_dir.iterdir()
                if path.is_dir() and re.fullmatch(r"\d{3}", path.name)
            ]
            for calculation_dir in calculation_dirs:
                structure = calculation_structure_file(calculation_dir)
                if structure:
                    discovered.append((structure, True, True))
            continue

        if not re.fullmatch(r"al_\d{3,}", round_dir.name):
            continue
        for role in ("train", "test"):
            role_dir = round_dir / role
            if not role_dir.is_dir():
                continue
            for calculation_dir in sorted(role_dir.iterdir()):
                if not (
                    calculation_dir.is_dir()
                    and re.fullmatch(r"\d{3}", calculation_dir.name)
                ):
                    continue
                structure = calculation_structure_file(calculation_dir)
                if structure:
                    # An active-learning output is selected/DFT-seen. It is
                    # model-seen only after it appears in a master train pool.
                    discovered.append((structure, False, True))
    return discovered


def discover_selected_round_trajectories(
    output_root: Path,
) -> list[tuple[Path, bool, bool]]:
    discovered: list[tuple[Path, bool, bool]] = []
    if not output_root.is_dir():
        return discovered
    for round_dir in sorted(output_root.iterdir()):
        if not (
            round_dir.is_dir()
            and re.fullmatch(r"al_\d{3,}", round_dir.name)
        ):
            continue
        for role in ("train", "test"):
            trajectory = round_dir / f"selected_{role}.traj"
            if trajectory.is_file() and trajectory.stat().st_size > 0:
                # Selected structures are already scheduled for DFT, but only
                # master-pool membership proves that a model has seen them.
                discovered.append((trajectory, False, True))
    return discovered


def load_reference_registry(
    model_pools: list[Path],
    dft_pools: list[Path],
    *,
    output_root: Path,
    scan_existing_rounds: bool,
    decimals: int,
) -> dict[str, ReferenceEntry]:
    registry: dict[str, ReferenceEntry] = {}
    path_roles: dict[Path, tuple[bool, bool]] = {}
    for path in dft_pools:
        path_roles[path] = (False, True)
    for path in model_pools:
        _, dft_seen = path_roles.get(path, (False, False))
        path_roles[path] = (True, dft_seen)

    for path, (model_seen, dft_seen) in path_roles.items():
        count_read = 0
        count_before = len(registry)
        for atoms in iread(str(path), index=":"):
            count_read += 1
            register_reference(
                registry,
                atoms,
                model_seen=model_seen,
                dft_seen=dft_seen,
                source=str(path),
                decimals=decimals,
            )
        print(
            f"Reference pool {path.name}: {count_read} rows, "
            f"{len(registry) - count_before} new unique structures"
        )

    if scan_existing_rounds:
        structures = discover_existing_round_structures(output_root)
        for path, model_seen, dft_seen in structures:
            try:
                atoms = read(str(path), format="vasp")
            except Exception as exc:
                print(f"Warning: could not register {path}: {exc}")
                continue
            register_reference(
                registry,
                atoms,
                model_seen=model_seen,
                dft_seen=dft_seen,
                source=str(path),
                decimals=decimals,
            )
        if structures:
            print(f"Scanned {len(structures)} structures from existing rounds")
        trajectories = discover_selected_round_trajectories(output_root)
        selected_count = 0
        for path, model_seen, dft_seen in trajectories:
            try:
                selected_frames = iread(str(path), index=":")
                for frame_index, atoms in enumerate(selected_frames):
                    selected_count += 1
                    register_reference(
                        registry,
                        atoms,
                        model_seen=model_seen,
                        dft_seen=dft_seen,
                        source=f"{path}#{frame_index}",
                        decimals=decimals,
                    )
            except Exception as exc:
                print(f"Warning: could not register {path}: {exc}")
        if trajectories:
            print(
                f"Scanned {selected_count} frames from "
                f"{len(trajectories)} selected round trajectories"
            )
    return registry


def write_reference_registry(
    path: Path, registry: dict[str, ReferenceEntry]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "structure_hash",
                "model_seen",
                "dft_seen",
                "n_atoms",
                "formula",
                "periodic_soap_compatible",
                "sources",
            ),
        )
        writer.writeheader()
        for key, entry in sorted(registry.items()):
            writer.writerow(
                {
                    "structure_hash": key,
                    "model_seen": int(entry.model_seen),
                    "dft_seen": int(entry.dft_seen),
                    "n_atoms": len(entry.atoms),
                    "formula": entry.atoms.get_chemical_formula(),
                    "periodic_soap_compatible": int(
                        periodic_soap_compatible(entry.atoms)
                    ),
                    "sources": json.dumps(sorted(entry.sources)),
                }
            )


def deduplicate_candidates(
    frames: list,
    source_trajectories: list[str],
    source_indices: list[int],
    dft_seen_hashes: set[str],
    decimals: int,
) -> tuple[list, list[dict[str, object]]]:
    unique_frames = []
    records: list[dict[str, object]] = []
    seen: dict[str, tuple[str, int]] = {}
    for sampled_index, (atoms, source_trajectory, source_index) in enumerate(
        zip(frames, source_trajectories, source_indices)
    ):
        key = structure_hash(atoms, decimals=decimals)
        record: dict[str, object] = {
            "sampled_index": sampled_index,
            "source_trajectory": source_trajectory,
            "source_frame": source_index,
            "structure_hash": key,
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "candidate_index": "",
            "duplicate_of_source_trajectory": "",
            "duplicate_of_source_frame": "",
            "status": "",
            "dataset_role": "",
            "selected_directory": "",
            "selection_rank": "",
            "role_rank": "",
        }
        if key in seen:
            record["status"] = "duplicate_candidate"
            duplicate_trajectory, duplicate_frame = seen[key]
            record["duplicate_of_source_trajectory"] = duplicate_trajectory
            record["duplicate_of_source_frame"] = duplicate_frame
        elif key in dft_seen_hashes:
            record["status"] = "already_dft_seen"
            seen[key] = (source_trajectory, source_index)
        else:
            record["status"] = "candidate"
            record["candidate_index"] = len(unique_frames)
            seen[key] = (source_trajectory, source_index)
            unique_frames.append(atoms.copy())
        records.append(record)
    return unique_frames, records


def evaluate_committee(
    frames: list,
    models: list[Path],
    device: str,
    default_dtype: str | None,
    calculator: str,
    use_kokkos: bool,
) -> dict[str, object]:
    n_frames = len(frames)
    n_models = len(models)
    energies = np.empty((n_frames, n_models), dtype=float)
    force_mean = [
        np.zeros((len(atoms), 3), dtype=float) for atoms in frames
    ]
    force_m2 = [np.zeros_like(value) for value in force_mean]

    for model_index, model in enumerate(models):
        backend = resolve_calculator_backend(model, calculator)
        calc = build_calculator(
            model,
            backend=backend,
            device=device,
            default_dtype=default_dtype,
            use_kokkos=use_kokkos,
        )
        print(
            f"Evaluating committee member {model_index + 1}/{n_models} "
            f"with {backend}: {model.name}"
        )
        for frame_index, atoms in enumerate(frames):
            work = atoms.copy()
            work.calc = calc
            energies[frame_index, model_index] = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), dtype=float)
            delta = forces - force_mean[frame_index]
            force_mean[frame_index] += delta / (model_index + 1)
            force_m2[frame_index] += delta * (
                forces - force_mean[frame_index]
            )
            if (frame_index + 1) % 50 == 0 or frame_index + 1 == n_frames:
                print(f"  evaluated {frame_index + 1}/{n_frames} frames")
        release_calculator(calc, device)

    energy_per_atom = energies / np.asarray(
        [len(atoms) for atoms in frames], dtype=float
    )[:, None]
    force_component_std = [
        np.sqrt(value / n_models) for value in force_m2
    ]
    force_std_rms = np.asarray(
        [
            math.sqrt(float(np.mean(np.sum(std**2, axis=1))))
            for std in force_component_std
        ]
    )
    force_std_max = np.asarray(
        [
            float(np.max(np.linalg.norm(std, axis=1)))
            for std in force_component_std
        ]
    )
    return {
        "energies": energies,
        "energy_mean": np.mean(energies, axis=1),
        "energy_std": np.std(energies, axis=1),
        "energy_per_atom_mean": np.mean(energy_per_atom, axis=1),
        "energy_per_atom_std": np.std(energy_per_atom, axis=1),
        "force_mean": force_mean,
        "force_component_std": force_component_std,
        "force_std_rms": force_std_rms,
        "force_std_max": force_std_max,
    }


def percentile_scores(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / (len(values) - 1)


def add_committee_metrics(
    records: list[dict[str, object]],
    metrics: dict[str, object],
    models: list[Path],
    energy_threshold_mev_atom: float,
    force_threshold_ev_a: float,
) -> list[int]:
    energies = np.asarray(metrics["energies"])
    energy_std = np.asarray(metrics["energy_per_atom_std"])
    force_std = np.asarray(metrics["force_std_rms"])
    energy_rank = percentile_scores(energy_std)
    force_rank = percentile_scores(force_std)
    uncertainty_score = np.maximum(energy_rank, force_rank)
    labels = model_labels(models)
    eligible_indices = []

    for record in records:
        if record["status"] != "candidate":
            continue
        index = int(record["candidate_index"])
        passes = (
            energy_threshold_mev_atom <= 0.0 and force_threshold_ev_a <= 0.0
        )
        if energy_threshold_mev_atom > 0.0:
            passes = passes or energy_std[index] * 1000.0 >= energy_threshold_mev_atom
        if force_threshold_ev_a > 0.0:
            passes = passes or force_std[index] >= force_threshold_ev_a
        record.update(
            {
                "committee_energy_mean_eV": float(
                    np.asarray(metrics["energy_mean"])[index]
                ),
                "committee_energy_std_eV": float(
                    np.asarray(metrics["energy_std"])[index]
                ),
                "committee_energy_mean_eV_atom": float(
                    np.asarray(metrics["energy_per_atom_mean"])[index]
                ),
                "committee_energy_std_meV_atom": float(
                    energy_std[index] * 1000.0
                ),
                "committee_force_std_rms_eV_A": float(force_std[index]),
                "committee_force_std_max_eV_A": float(
                    np.asarray(metrics["force_std_max"])[index]
                ),
                "uncertainty_percentile": float(uncertainty_score[index]),
            }
        )
        for model_index, label in enumerate(labels):
            record[f"{label}_energy_eV"] = float(energies[index, model_index])
        if passes:
            record["status"] = "eligible"
            eligible_indices.append(index)
        else:
            record["status"] = "below_uncertainty_threshold"
    return eligible_indices


def soap_descriptors(
    frames: list,
    species: list[str],
    r_cut: float,
    n_max: int,
    l_max: int,
    n_jobs: int,
) -> np.ndarray:
    from dscribe.descriptors import SOAP

    incompatible = [
        (
            index,
            atoms.get_chemical_formula(),
            int(atoms.cell.rank),
            atoms.get_pbc().tolist(),
        )
        for index, atoms in enumerate(frames)
        if not periodic_soap_compatible(atoms)
    ]
    if incompatible:
        preview = "; ".join(
            f"index={index} formula={formula} cell_rank={rank} pbc={pbc}"
            for index, formula, rank, pbc in incompatible[:5]
        )
        raise ValueError(
            "Periodic SOAP requires a full-rank periodic cell for every "
            f"candidate structure. Incompatible structures: {preview}"
        )
    missing_species = sorted(
        {symbol for atoms in frames for symbol in atoms.symbols} - set(species)
    )
    if missing_species:
        raise ValueError(
            "SOAP species list does not include: "
            + ", ".join(missing_species)
            + ". Extend --soap-species."
        )
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",
        sparse=False,
    )
    descriptors = np.asarray(
        soap.create(frames, n_jobs=n_jobs), dtype=np.float32
    )
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return np.asarray(
        descriptors / np.maximum(norms, 1.0e-15), dtype=np.float32
    )


def soap_cache_settings(
    *,
    species: list[str],
    r_cut: float,
    n_max: int,
    l_max: int,
    hash_decimals: int,
) -> dict[str, object]:
    try:
        dscribe_version = importlib.metadata.version("dscribe")
    except importlib.metadata.PackageNotFoundError:
        dscribe_version = "unknown"
    return {
        "species": species,
        "r_cut": float(r_cut),
        "n_max": int(n_max),
        "l_max": int(l_max),
        "periodic": True,
        "average": "inner",
        "dtype": "float32",
        "hash_decimals": int(hash_decimals),
        "dscribe_version": dscribe_version,
    }


def cached_reference_soap_descriptors(
    cache_path: Path,
    registry: dict[str, ReferenceEntry],
    *,
    settings: dict[str, object],
    n_jobs: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    import fcntl
    import h5py

    compatible_registry = {
        key: entry
        for key, entry in registry.items()
        if periodic_soap_compatible(entry.atoms)
    }
    skipped = len(registry) - len(compatible_registry)
    if skipped:
        print(
            f"Skipping {skipped} nonperiodic or cell-less reference "
            "structures for periodic SOAP"
        )
    if not compatible_registry:
        return {}, {"hits": 0, "misses": 0, "skipped": skipped, "stored": 0}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    requested_hashes = sorted(compatible_registry)
    settings_json = json.dumps(settings, sort_keys=True, separators=(",", ":"))

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with h5py.File(cache_path, "a") as handle:
            stored_settings = str(handle.attrs.get("settings_json", ""))
            if stored_settings and stored_settings != settings_json:
                print("SOAP settings changed; rebuilding the reference cache")
                for name in list(handle.keys()):
                    del handle[name]
            handle.attrs["settings_json"] = settings_json

            if "hashes" in handle:
                raw_hashes = handle["hashes"][:]
                stored_hashes = [
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                    for value in raw_hashes
                ]
            else:
                stored_hashes = []
            index_by_hash = {
                key: index for index, key in enumerate(stored_hashes)
            }
            missing_hashes = [
                key for key in requested_hashes if key not in index_by_hash
            ]
            hits = len(requested_hashes) - len(missing_hashes)

            if missing_hashes:
                print(
                    f"SOAP reference cache: {hits} hits, "
                    f"{len(missing_hashes)} descriptors to calculate"
                )
                missing_frames = [
                    compatible_registry[key].atoms for key in missing_hashes
                ]
                new_descriptors = soap_descriptors(
                    missing_frames,
                    species=list(settings["species"]),
                    r_cut=float(settings["r_cut"]),
                    n_max=int(settings["n_max"]),
                    l_max=int(settings["l_max"]),
                    n_jobs=n_jobs,
                )
                if "hashes" not in handle:
                    string_dtype = h5py.string_dtype(encoding="utf-8")
                    hash_dataset = handle.create_dataset(
                        "hashes",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=string_dtype,
                    )
                    descriptor_dataset = handle.create_dataset(
                        "descriptors",
                        shape=(0, new_descriptors.shape[1]),
                        maxshape=(None, new_descriptors.shape[1]),
                        dtype=np.float32,
                    )
                else:
                    hash_dataset = handle["hashes"]
                    descriptor_dataset = handle["descriptors"]
                    if descriptor_dataset.shape[1] != new_descriptors.shape[1]:
                        raise ValueError(
                            "SOAP cache descriptor width does not match its settings"
                        )
                start = len(hash_dataset)
                stop = start + len(missing_hashes)
                hash_dataset.resize((stop,))
                descriptor_dataset.resize((stop, descriptor_dataset.shape[1]))
                hash_dataset[start:stop] = missing_hashes
                descriptor_dataset[start:stop] = new_descriptors
                handle.flush()
                stored_hashes.extend(missing_hashes)
                index_by_hash.update(
                    {
                        key: start + offset
                        for offset, key in enumerate(missing_hashes)
                    }
                )
            else:
                print(
                    f"SOAP reference cache: {hits} hits, no descriptors to calculate"
                )

            all_descriptors = np.asarray(
                handle["descriptors"][:], dtype=np.float32
            )
            descriptors = {
                key: all_descriptors[index_by_hash[key]]
                for key in requested_hashes
            }
            stats = {
                "hits": hits,
                "misses": len(missing_hashes),
                "skipped": skipped,
                "stored": len(stored_hashes),
            }
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return descriptors, stats


def store_selected_soap_descriptors(
    cache_path: Path,
    descriptors_by_hash: dict[str, np.ndarray],
    *,
    settings: dict[str, object],
) -> int:
    """Persist already-computed selected descriptors for subsequent rounds."""
    import fcntl
    import h5py

    if not descriptors_by_hash:
        return 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(f".{cache_path.name}.lock")
    settings_json = json.dumps(settings, sort_keys=True, separators=(",", ":"))

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with h5py.File(cache_path, "a") as handle:
            stored_settings = str(handle.attrs.get("settings_json", ""))
            if stored_settings and stored_settings != settings_json:
                raise RuntimeError(
                    "SOAP cache settings changed during a selection run"
                )
            handle.attrs["settings_json"] = settings_json
            if "hashes" in handle:
                stored_hashes = {
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                    for value in handle["hashes"][:]
                }
            else:
                stored_hashes = set()
            missing_hashes = sorted(set(descriptors_by_hash) - stored_hashes)
            if not missing_hashes:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                return 0

            descriptor_width = len(descriptors_by_hash[missing_hashes[0]])
            if "hashes" not in handle:
                string_dtype = h5py.string_dtype(encoding="utf-8")
                hash_dataset = handle.create_dataset(
                    "hashes",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=string_dtype,
                )
                descriptor_dataset = handle.create_dataset(
                    "descriptors",
                    shape=(0, descriptor_width),
                    maxshape=(None, descriptor_width),
                    dtype=np.float32,
                )
            else:
                hash_dataset = handle["hashes"]
                descriptor_dataset = handle["descriptors"]
                if descriptor_dataset.shape[1] != descriptor_width:
                    raise ValueError(
                        "Selected SOAP descriptor width does not match the cache"
                    )
            start = len(hash_dataset)
            stop = start + len(missing_hashes)
            hash_dataset.resize((stop,))
            descriptor_dataset.resize((stop, descriptor_width))
            hash_dataset[start:stop] = missing_hashes
            descriptor_dataset[start:stop] = np.asarray(
                [descriptors_by_hash[key] for key in missing_hashes],
                dtype=np.float32,
            )
            handle.flush()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    print(f"Stored {len(missing_hashes)} selected descriptors in the SOAP cache")
    return len(missing_hashes)


def nearest_soap_distances(
    candidate_descriptors: np.ndarray, reference_descriptors: np.ndarray
) -> np.ndarray:
    if len(reference_descriptors) == 0:
        return np.full(len(candidate_descriptors), np.nan, dtype=float)
    return np.min(
        cdist(candidate_descriptors, reference_descriptors, metric="euclidean"),
        axis=1,
    )


def priority_aware_maxmin(
    descriptors: np.ndarray,
    priority_scores: np.ndarray,
    n_select: int,
) -> list[int]:
    if len(descriptors) == 0 or n_select <= 0:
        return []
    n_select = min(n_select, len(descriptors))
    if len(descriptors) == 1:
        return [0]
    distances = cdist(descriptors, descriptors, metric="euclidean")
    selected = [int(np.argmax(priority_scores))]
    while len(selected) < n_select:
        remaining = np.asarray(
            [index for index in range(len(descriptors)) if index not in selected],
            dtype=int,
        )
        min_distance = np.min(distances[remaining][:, selected], axis=1)
        if float(np.max(min_distance)) > 0.0:
            diversity = min_distance / float(np.max(min_distance))
        else:
            diversity = np.ones_like(min_distance)
        combined = diversity * (0.5 + 0.5 * priority_scores[remaining])
        selected.append(int(remaining[int(np.argmax(combined))]))
    return selected


def representative_test_positions(
    descriptors: np.ndarray, n_test: int
) -> list[int]:
    if n_test <= 0 or len(descriptors) == 0:
        return []
    n_test = min(n_test, len(descriptors))
    if len(descriptors) == 1:
        return [0]
    distances = cdist(descriptors, descriptors, metric="euclidean")
    selected = [int(np.argmin(np.mean(distances, axis=1)))]
    while len(selected) < n_test:
        remaining = np.asarray(
            [index for index in range(len(descriptors)) if index not in selected],
            dtype=int,
        )
        min_distance = np.min(distances[remaining][:, selected], axis=1)
        selected.append(int(remaining[int(np.argmax(min_distance))]))
    return selected


def write_selected_trajectory(
    path: Path,
    selected_candidate_indices: list[int],
    frames: list,
    metrics: dict[str, object],
    records_by_candidate: dict[int, dict[str, object]],
    model_paths: list[Path],
    dataset_role: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with AseTrajectory(str(path), "w") as writer:
        for candidate_index in selected_candidate_indices:
            atoms = frames[candidate_index].copy()
            atoms.set_constraint(None)
            record = records_by_candidate[candidate_index]
            atoms.info.update(
                {
                    "source_trajectory": str(record["source_trajectory"]),
                    "source_frame": int(record["source_frame"]),
                    "candidate_index": candidate_index,
                    "dataset_role": dataset_role,
                    "selection_rank": int(record["selection_rank"]),
                    "role_rank": int(record["role_rank"]),
                    "committee_models": json.dumps(
                        [str(model) for model in model_paths]
                    ),
                    "committee_energy_std_meV_atom": float(
                        np.asarray(metrics["energy_per_atom_std"])[candidate_index]
                        * 1000.0
                    ),
                    "committee_force_std_rms_eV_A": float(
                        np.asarray(metrics["force_std_rms"])[candidate_index]
                    ),
                }
            )
            atoms.arrays["committee_force_std"] = np.asarray(
                metrics["force_component_std"][candidate_index]
            )
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(np.asarray(metrics["energy_mean"])[candidate_index]),
                forces=np.asarray(metrics["force_mean"][candidate_index]),
            )
            writer.write(atoms)


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "sampled_index",
        "source_trajectory",
        "source_frame",
        "structure_hash",
        "n_atoms",
        "formula",
        "candidate_index",
        "duplicate_of_source_trajectory",
        "duplicate_of_source_frame",
        "status",
        "dataset_role",
        "selected_directory",
        "selection_rank",
        "role_rank",
        "committee_energy_mean_eV",
        "committee_energy_std_eV",
        "committee_energy_mean_eV_atom",
        "committee_energy_std_meV_atom",
        "committee_force_std_rms_eV_A",
        "committee_force_std_max_eV_A",
        "uncertainty_percentile",
        "nearest_model_seen_soap_distance",
        "nearest_dft_seen_soap_distance",
        "soap_novelty_percentile",
        "selection_priority",
        "nearest_train_soap_distance",
    ]
    all_fields = {key for record in records for key in record}
    fieldnames = preferred + sorted(all_fields - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_soap_species(value: str) -> list[str]:
    species = [item.strip() for item in value.split(",") if item.strip()]
    if not species:
        raise ValueError("--soap-species must contain at least one element")
    if len(species) != len(set(species)):
        raise ValueError("--soap-species contains duplicate elements")
    return species


def existing_round_numbers(output_root: Path) -> list[int]:
    numbers: list[int] = []
    if not output_root.is_dir():
        return numbers
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"al_(\d{3,})", path.name)
        if match:
            numbers.append(int(match.group(1)))
            continue
        if not path.name.isdigit():
            continue
        has_calculations = any(
            child.is_dir()
            and re.fullmatch(r"\d{3}", child.name)
            and calculation_structure_file(child) is not None
            for child in path.iterdir()
        )
        if has_calculations:
            numbers.append(int(path.name))
    return numbers


def resolve_round_number(value: str, output_root: Path) -> int:
    if value.lower() == "auto":
        existing = existing_round_numbers(output_root)
        return max(existing, default=-1) + 1
    match = re.fullmatch(r"(?:al_)?(\d+)", value)
    if not match:
        raise ValueError("--round must be 'auto', an integer, or al_NNN")
    return int(match.group(1))


def write_round_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_directory_role(directory: Path, role: str, round_number: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_round_metadata(
        directory / "dataset_role.json",
        {
            "dataset_role": role,
            "active_learning_round": round_number,
            "include_in_mace_finetuning": role == "train",
        },
    )
    if role == "test":
        (directory / "EXCLUDE_FROM_MACE_FINETUNING").write_text(
            "This external-test configuration must never enter MACE training.\n",
            encoding="utf-8",
        )


def write_directory_list(path: Path, directories: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{directory}\n" for directory in directories),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcmc-active-learning-select",
        description=(
            "Select novel, uncertain, and SOAP-diverse trajectory frames with "
            "a model committee, then prepare static VASP calculations."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        help=(
            "XDATCAR or ASE-readable trajectory. Repeat to merge multiple "
            "trajectory pools; auto-detected when omitted."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--train-size",
        type=int,
        default=8,
        help="Number of configurations written to al_NNN/train.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=4,
        help="Maximum configurations written to al_NNN/test.",
    )
    parser.add_argument(
        "--n-select",
        type=int,
        default=None,
        help="Number selected with --legacy-flat-output (default: train-size).",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=300,
        help=(
            "Uncertainty-ranked pool passed to SOAP. "
            "0 uses ten times the requested total selection."
        ),
    )
    add_model_arguments(parser, default_device="cpu")
    parser.add_argument(
        "--reference-pool",
        action="append",
        help=(
            "Legacy alias: add a pool as both model-seen and DFT-seen."
        ),
    )
    parser.add_argument(
        "--model-seen-pool",
        action="append",
        help="Additional ASE-readable structures seen during MACE fine-tuning.",
    )
    parser.add_argument(
        "--dft-seen-pool",
        action="append",
        help="Additional ASE-readable structures already evaluated with DFT.",
    )
    parser.add_argument(
        "--no-reference-pool",
        action="store_true",
        help=(
            "Disable the default training/test pools. Explicit model-seen, "
            "DFT-seen, and legacy reference pools are still used."
        ),
    )
    parser.add_argument(
        "--no-scan-existing-rounds",
        action="store_true",
        help="Do not register legacy numeric or al_NNN rounds below output-root.",
    )
    parser.add_argument("--hash-decimals", type=int, default=6)
    parser.add_argument(
        "--energy-std-threshold-mev-atom",
        type=float,
        default=0.0,
        help="Optional committee energy threshold. Candidates pass if either active threshold passes.",
    )
    parser.add_argument(
        "--force-std-threshold-ev-a",
        type=float,
        default=0.0,
        help="Optional RMS force-disagreement threshold. Zero disables it.",
    )
    parser.add_argument("--soap-r-cut", type=float, default=6.0)
    parser.add_argument("--soap-n-max", type=int, default=8)
    parser.add_argument("--soap-l-max", type=int, default=6)
    parser.add_argument("--soap-n-jobs", type=int, default=1)
    parser.add_argument(
        "--soap-species",
        default=",".join(DEFAULT_SOAP_SPECIES),
        help="Comma-separated fixed SOAP species list.",
    )
    parser.add_argument(
        "--soap-cache",
        type=Path,
        default=None,
        help="HDF5 descriptor cache. Defaults to output-root/.active_learning/.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Reference registry CSV. Defaults to output-root/.active_learning/.",
    )
    parser.add_argument(
        "--novelty-weight",
        type=float,
        default=0.45,
        help="Weight of model-pool SOAP novelty in selection priority.",
    )
    parser.add_argument(
        "--min-model-soap-distance",
        type=float,
        default=0.0,
        help="Optional hard minimum distance from model-seen structures.",
    )
    parser.add_argument(
        "--min-dft-soap-distance",
        type=float,
        default=0.0,
        help="Optional hard minimum distance from all DFT-seen structures.",
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--round",
        default="auto",
        help="Active-learning round number or 'auto'. Output is al_NNN.",
    )
    parser.add_argument(
        "--legacy-flat-output",
        action="store_true",
        help="Write the previous flat 000, 001, ... layout without a test split.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--functional",
        choices=("PBE0-rVV10_8", "PBE-D3"),
        default="PBE0-rVV10_8",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path. Defaults inside al_NNN (or cwd in legacy mode).",
    )
    parser.add_argument(
        "--selected-traj",
        type=Path,
        default=None,
        help="Legacy selected trajectory path; active layout writes train/test trajectories.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the manifest/trajectory without creating VASP directories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_size < 1:
        raise ValueError("--train-size must be at least 1")
    if args.test_size < 0:
        raise ValueError("--test-size cannot be negative")
    if not 0.0 <= args.novelty_weight <= 1.0:
        raise ValueError("--novelty-weight must be between 0 and 1")
    if args.min_model_soap_distance < 0.0:
        raise ValueError("--min-model-soap-distance cannot be negative")
    if args.min_dft_soap_distance < 0.0:
        raise ValueError("--min-dft-soap-distance cannot be negative")

    output_root = args.output_root.expanduser().resolve()
    if args.legacy_flat_output:
        n_train = args.n_select if args.n_select is not None else args.train_size
        if n_train < 1:
            raise ValueError("--n-select must be at least 1")
        n_test = 0
        round_number = None
        round_root = output_root
        manifest = (
            args.manifest.expanduser().resolve()
            if args.manifest
            else (Path.cwd() / "active_learning_candidates.csv").resolve()
        )
        train_trajectory = (
            args.selected_traj.expanduser().resolve()
            if args.selected_traj
            else (Path.cwd() / "selected_frames.traj").resolve()
        )
        test_trajectory = None
    else:
        if args.n_select is not None:
            raise ValueError("--n-select is only valid with --legacy-flat-output")
        n_train = args.train_size
        n_test = args.test_size
        round_number = resolve_round_number(str(args.round), output_root)
        round_root = output_root / f"al_{round_number:03d}"
        if round_root.exists():
            raise FileExistsError(
                f"{round_root} already exists; choose another --round"
            )
        manifest = (
            args.manifest.expanduser().resolve()
            if args.manifest
            else round_root / "selection.csv"
        )
        train_trajectory = round_root / "selected_train.traj"
        test_trajectory = round_root / "selected_test.traj"

    target_total = n_train + n_test
    input_paths = resolve_input_trajectories(args.input)
    models = resolve_models(
        args.model_path,
        args.calculator,
        minimum=2,
    )
    model_pools, dft_pools = resolve_reference_pools(
        models,
        args.model_seen_pool,
        args.dft_seen_pool,
        args.reference_pool,
        args.no_reference_pool,
    )
    registry = load_reference_registry(
        model_pools,
        dft_pools,
        output_root=output_root,
        scan_existing_rounds=not args.no_scan_existing_rounds,
        decimals=args.hash_decimals,
    )
    model_seen_hashes = {
        key for key, entry in registry.items() if entry.model_seen
    }
    dft_seen_hashes = {
        key for key, entry in registry.items() if entry.dft_seen
    }
    print(
        f"Reference registry: {len(model_seen_hashes)} model-seen, "
        f"{len(dft_seen_hashes)} DFT-seen unique structures"
    )

    active_learning_dir = output_root / ".active_learning"
    registry_path = (
        args.registry.expanduser().resolve()
        if args.registry
        else active_learning_dir / "structure_registry.csv"
    )
    cache_path = (
        args.soap_cache.expanduser().resolve()
        if args.soap_cache
        else active_learning_dir / "soap_reference_cache.h5"
    )
    write_reference_registry(registry_path, registry)
    species = parse_soap_species(args.soap_species)
    settings = soap_cache_settings(
        species=species,
        r_cut=args.soap_r_cut,
        n_max=args.soap_n_max,
        l_max=args.soap_l_max,
        hash_decimals=args.hash_decimals,
    )
    reference_descriptors, cache_stats = cached_reference_soap_descriptors(
        cache_path,
        registry,
        settings=settings,
        n_jobs=args.soap_n_jobs,
    )

    frames = []
    source_indices = []
    source_trajectories = []
    for input_path in input_paths:
        input_frames, input_source_indices = load_trajectory_frames(
            input_path, args.frame_stride
        )
        frames.extend(input_frames)
        source_indices.extend(input_source_indices)
        source_trajectories.extend([str(input_path)] * len(input_frames))
    print(
        f"Combined input pool: {len(frames)} sampled frames from "
        f"{len(input_paths)} trajectories"
    )
    candidates, records = deduplicate_candidates(
        frames,
        source_trajectories,
        source_indices,
        dft_seen_hashes,
        args.hash_decimals,
    )
    print(
        f"Screening pool: {len(candidates)} novel unique frames "
        f"from {len(frames)} sampled frames"
    )
    if not candidates:
        write_manifest(manifest, records)
        raise ValueError("No novel trajectory frames remain after deduplication.")

    metrics = evaluate_committee(
        candidates,
        models=models,
        device=args.device,
        default_dtype=args.default_dtype,
        calculator=args.calculator,
        use_kokkos=args.use_kokkos,
    )
    eligible = add_committee_metrics(
        records,
        metrics,
        models,
        energy_threshold_mev_atom=args.energy_std_threshold_mev_atom,
        force_threshold_ev_a=args.force_std_threshold_ev_a,
    )
    if not eligible:
        write_manifest(manifest, records)
        raise ValueError(
            "No candidates pass the uncertainty thresholds. "
            "Lower a threshold or set both thresholds to zero."
        )

    record_by_candidate = {
        int(record["candidate_index"]): record
        for record in records
        if record.get("candidate_index") != ""
    }
    uncertainty = np.asarray(
        [
            float(record_by_candidate[candidate_index]["uncertainty_percentile"])
            for candidate_index in eligible
        ]
    )
    pool_size = args.candidate_pool_size or max(target_total * 10, target_total)
    pool_order = np.argsort(-uncertainty, kind="stable")[:pool_size]
    pool_candidate_indices = [eligible[index] for index in pool_order]
    pool_frames = [candidates[index] for index in pool_candidate_indices]
    pool_uncertainty = uncertainty[pool_order]
    pool_descriptors = soap_descriptors(
        pool_frames,
        species=species,
        r_cut=args.soap_r_cut,
        n_max=args.soap_n_max,
        l_max=args.soap_l_max,
        n_jobs=args.soap_n_jobs,
    )

    descriptor_width = pool_descriptors.shape[1]
    model_reference_matrix = np.asarray(
        [
            reference_descriptors[key]
            for key in sorted(model_seen_hashes)
            if key in reference_descriptors
        ],
        dtype=np.float32,
    )
    dft_reference_matrix = np.asarray(
        [
            reference_descriptors[key]
            for key in sorted(dft_seen_hashes)
            if key in reference_descriptors
        ],
        dtype=np.float32,
    )
    if model_reference_matrix.size == 0:
        model_reference_matrix = np.empty((0, descriptor_width), dtype=np.float32)
    if dft_reference_matrix.size == 0:
        dft_reference_matrix = np.empty((0, descriptor_width), dtype=np.float32)
    model_distances = nearest_soap_distances(
        pool_descriptors, model_reference_matrix
    )
    dft_distances = nearest_soap_distances(
        pool_descriptors, dft_reference_matrix
    )
    if np.all(np.isfinite(model_distances)):
        novelty_percentile = percentile_scores(model_distances)
    else:
        novelty_percentile = np.zeros(len(pool_descriptors), dtype=float)
    priority = (
        (1.0 - args.novelty_weight) * pool_uncertainty
        + args.novelty_weight * novelty_percentile
    )

    valid_pool_positions = []
    for pool_position, candidate_index in enumerate(pool_candidate_indices):
        record = record_by_candidate[candidate_index]
        record.update(
            {
                "nearest_model_seen_soap_distance": float(
                    model_distances[pool_position]
                ),
                "nearest_dft_seen_soap_distance": float(
                    dft_distances[pool_position]
                ),
                "soap_novelty_percentile": float(
                    novelty_percentile[pool_position]
                ),
                "selection_priority": float(priority[pool_position]),
            }
        )
        if (
            args.min_dft_soap_distance > 0.0
            and math.isfinite(dft_distances[pool_position])
            and dft_distances[pool_position] < args.min_dft_soap_distance
        ):
            record["status"] = "too_close_to_dft_pool"
        elif (
            args.min_model_soap_distance > 0.0
            and math.isfinite(model_distances[pool_position])
            and model_distances[pool_position] < args.min_model_soap_distance
        ):
            record["status"] = "too_close_to_model_pool"
        else:
            valid_pool_positions.append(pool_position)

    if len(valid_pool_positions) < n_train:
        write_manifest(manifest, records)
        raise ValueError(
            f"Only {len(valid_pool_positions)} candidates remain after SOAP "
            f"screening; {n_train} training structures are required."
        )

    valid_descriptors = pool_descriptors[valid_pool_positions]
    valid_priority = priority[valid_pool_positions]
    selected_valid_positions = priority_aware_maxmin(
        valid_descriptors,
        valid_priority,
        n_select=min(target_total, len(valid_pool_positions)),
    )
    selected_pool_positions = [
        valid_pool_positions[index] for index in selected_valid_positions
    ]
    selected = [
        pool_candidate_indices[index] for index in selected_pool_positions
    ]
    actual_test_size = min(n_test, max(0, len(selected) - n_train))
    selected_descriptors = pool_descriptors[selected_pool_positions]
    test_selected_positions = set(
        representative_test_positions(selected_descriptors, actual_test_size)
    )
    test_selected = [
        candidate_index
        for position, candidate_index in enumerate(selected)
        if position in test_selected_positions
    ]
    train_selected = [
        candidate_index
        for position, candidate_index in enumerate(selected)
        if position not in test_selected_positions
    ]
    if len(train_selected) != n_train:
        raise RuntimeError(
            f"Internal split error: selected {len(train_selected)} training "
            f"structures, expected {n_train}"
        )
    if len(test_selected) < n_test:
        print(
            f"Warning: selected {len(test_selected)} test structures "
            f"instead of the requested {n_test}"
        )

    descriptor_by_candidate = {
        candidate_index: pool_descriptors[pool_position]
        for pool_position, candidate_index in enumerate(pool_candidate_indices)
    }
    selected_descriptor_map = {
        str(record_by_candidate[index]["structure_hash"]): descriptor_by_candidate[
            index
        ]
        for index in train_selected + test_selected
    }
    selected_cache_additions = store_selected_soap_descriptors(
        cache_path,
        selected_descriptor_map,
        settings=settings,
    )
    cache_stats["selected_added"] = selected_cache_additions
    cache_stats["stored_after_selection"] = (
        cache_stats["stored"] + selected_cache_additions
    )
    if train_selected and test_selected:
        train_matrix = np.asarray(
            [descriptor_by_candidate[index] for index in train_selected]
        )
        test_matrix = np.asarray(
            [descriptor_by_candidate[index] for index in test_selected]
        )
        nearest_train = nearest_soap_distances(test_matrix, train_matrix)
        for candidate_index, distance in zip(test_selected, nearest_train):
            record_by_candidate[candidate_index][
                "nearest_train_soap_distance"
            ] = float(distance)

    pool_set = set(pool_candidate_indices)
    selected_set = set(selected)
    for candidate_index in eligible:
        record = record_by_candidate[candidate_index]
        if candidate_index in selected_set:
            record["status"] = "selected"
        elif candidate_index not in pool_set:
            record["status"] = "eligible_not_shortlisted"
        elif record["status"] == "eligible":
            record["status"] = "soap_pool_not_selected"

    for selection_rank, candidate_index in enumerate(selected):
        record = record_by_candidate[candidate_index]
        record["selection_rank"] = selection_rank
    for role_rank, candidate_index in enumerate(train_selected):
        directory = (
            output_root / f"{args.start_index + role_rank:03d}"
            if args.legacy_flat_output
            else round_root / "train" / f"{role_rank:03d}"
        )
        record = record_by_candidate[candidate_index]
        record["dataset_role"] = "train"
        record["role_rank"] = role_rank
        record["selected_directory"] = str(directory)
    for role_rank, candidate_index in enumerate(test_selected):
        directory = round_root / "test" / f"{role_rank:03d}"
        record = record_by_candidate[candidate_index]
        record["dataset_role"] = "test"
        record["role_rank"] = role_rank
        record["selected_directory"] = str(directory)

    write_selected_trajectory(
        train_trajectory,
        train_selected,
        candidates,
        metrics,
        record_by_candidate,
        models,
        "train",
    )
    if test_selected and test_trajectory is not None:
        write_selected_trajectory(
            test_trajectory,
            test_selected,
            candidates,
            metrics,
            record_by_candidate,
            models,
            "test",
        )
    write_manifest(manifest, records)

    if not args.legacy_flat_output:
        train_directories = [
            Path(str(record_by_candidate[index]["selected_directory"]))
            for index in train_selected
        ]
        test_directories = [
            Path(str(record_by_candidate[index]["selected_directory"]))
            for index in test_selected
        ]
        write_directory_list(
            round_root / "train_directories.txt", train_directories
        )
        write_directory_list(
            round_root / "test_directories.txt", test_directories
        )
        write_round_metadata(
            round_root / "round.json",
            {
                "round": round_number,
                "inputs": [str(path) for path in input_paths],
                "models": [str(model) for model in models],
                "model_backends": [
                    resolve_calculator_backend(model, args.calculator)
                    for model in models
                ],
                "train_size": len(train_selected),
                "test_size": len(test_selected),
                "train_hashes": [
                    record_by_candidate[index]["structure_hash"]
                    for index in train_selected
                ],
                "test_hashes": [
                    record_by_candidate[index]["structure_hash"]
                    for index in test_selected
                ],
                "soap_settings": settings,
                "soap_cache": str(cache_path),
                "soap_cache_stats": cache_stats,
                "registry": str(registry_path),
                "model_seen_count": len(model_seen_hashes),
                "dft_seen_count": len(dft_seen_hashes),
                "novelty_weight": args.novelty_weight,
                "test_policy": "never_include_in_mace_finetuning",
            },
        )

    for candidate_index in train_selected + test_selected:
        record = record_by_candidate[candidate_index]
        directory = Path(
            str(record["selected_directory"])
        )
        if not args.selection_only:
            from pymatgen.io.ase import AseAtomsAdaptor

            if round_number is not None:
                write_directory_role(
                    directory,
                    str(record["dataset_role"]),
                    round_number,
                )
            atoms = candidates[candidate_index].copy()
            atoms.set_constraint(None)
            structure = AseAtomsAdaptor.get_structure(atoms)
            Generate(structure, directory, args.functional).scf(
                overwrite=args.overwrite
            )
    print(
        f"Selected {len(train_selected)} training and {len(test_selected)} "
        f"test structures from {len(eligible)} eligible frames"
    )
    print(f"Manifest: {manifest}")
    print(f"Training trajectory: {train_trajectory}")
    if test_selected and test_trajectory is not None:
        print(f"Test trajectory: {test_trajectory}")
    print(f"SOAP cache: {cache_path}")


if __name__ == "__main__":
    main()
