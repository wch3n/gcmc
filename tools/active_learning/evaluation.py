#!/usr/bin/env python3
"""Evaluate a model committee on active-learning VASP calculations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory

from .models import (
    add_model_arguments,
    build_calculator,
    model_labels,
    release_calculator,
    resolve_calculator_backend,
    resolve_models,
)


def parse_indices(text: str) -> list[str]:
    items: list[str] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_s, stop_s = token.split("-", 1)
            start = int(start_s)
            stop = int(stop_s)
            step = 1 if stop >= start else -1
            items.extend(f"{index:03d}" for index in range(start, stop + step, step))
        else:
            items.append(f"{int(token):03d}")
    return items


def natural_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    ]


def latest_oszicar_energy(path: Path) -> float:
    if not path.exists():
        return float("nan")
    energy = float("nan")
    pattern = re.compile(r"\bE0=\s*([-+0-9.Ee]+)")
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            energy = float(match.group(1))
    return energy


def find_structure_file(subdir: Path) -> Path | None:
    for name in ("CONTCAR", "POSCAR"):
        candidate = subdir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def iter_structure_dirs(
    root: Path, labels: Iterable[str], recursive: bool
) -> list[tuple[str, str, Path, Path | None]]:
    label_set = set(labels)
    directories: list[Path]
    if recursive:
        directories = [
            path
            for path in root.rglob("*")
            if path.is_dir() and path.name in label_set
        ]
    else:
        directories = [root / label for label in labels]
    directories.sort(key=lambda path: natural_key(str(path.relative_to(root))))

    entries = []
    for subdir in directories:
        relative = subdir.relative_to(root)
        series_path = relative.parent
        series = "." if str(series_path) == "." else str(series_path)
        entries.append(
            (str(relative), series, subdir, find_structure_file(subdir))
        )
    return entries


def read_dft_configuration(
    subdir: Path, structure: Path
) -> tuple[object, float, np.ndarray | None, str]:
    atoms = read(str(structure), format="vasp")
    energy = float("nan")
    forces = None
    source = ""

    for filename in ("vasprun.xml", "OUTCAR"):
        result_path = subdir / filename
        if not result_path.exists() or result_path.stat().st_size == 0:
            continue
        try:
            dft_atoms = read(str(result_path), index=-1)
            if len(dft_atoms) != len(atoms):
                raise ValueError(
                    f"{filename} has {len(dft_atoms)} atoms, expected {len(atoms)}"
                )
            energy = float(dft_atoms.get_potential_energy())
            forces = np.asarray(dft_atoms.get_forces(), dtype=float)
            atoms = dft_atoms
            source = filename
            break
        except Exception as exc:
            print(f"Warning: could not read {result_path}: {exc}")

    if not math.isfinite(energy):
        energy = latest_oszicar_energy(subdir / "OSZICAR")
        if math.isfinite(energy):
            source = "OSZICAR"
    return atoms, energy, forces, source


def force_error_stats(
    predicted: np.ndarray, reference: np.ndarray
) -> tuple[float, float, float]:
    delta = np.asarray(predicted) - np.asarray(reference)
    mae = float(np.mean(np.abs(delta)))
    rmse = float(math.sqrt(np.mean(delta**2)))
    vector_max = float(np.max(np.linalg.norm(delta, axis=1)))
    return mae, rmse, vector_max


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def metric_summary(records: list[dict[str, object]], labels: list[str]) -> dict:
    valid_energy = [
        record
        for record in records
        if math.isfinite(float(record.get("_dft_energy", float("nan"))))
    ]
    valid_forces = [
        record
        for record in records
        if math.isfinite(float(record.get("_ensemble_force_rmse", float("nan"))))
    ]
    ensemble_energy_errors = np.asarray(
        [
            float(record["_ensemble_energy_error_per_atom"])
            for record in valid_energy
        ],
        dtype=float,
    )
    summary: dict[str, object] = {
        "n_configurations": len(records),
        "n_with_dft_energy": len(valid_energy),
        "n_with_dft_forces": len(valid_forces),
        "ensemble_energy_mae_meV_atom": (
            float(np.mean(np.abs(ensemble_energy_errors)) * 1000.0)
            if len(ensemble_energy_errors)
            else float("nan")
        ),
        "ensemble_energy_rmse_meV_atom": (
            float(math.sqrt(np.mean(ensemble_energy_errors**2)) * 1000.0)
            if len(ensemble_energy_errors)
            else float("nan")
        ),
        "ensemble_force_mae_eV_A": finite_mean(
            float(record["_ensemble_force_mae"]) for record in valid_forces
        ),
        "ensemble_force_rmse_eV_A": (
            float(
                math.sqrt(
                    np.mean(
                        [
                            float(record["_ensemble_force_rmse"]) ** 2
                            for record in valid_forces
                        ]
                    )
                )
            )
            if valid_forces
            else float("nan")
        ),
    }
    for label in labels:
        model_energy_errors = np.asarray(
            [
                float(record[f"_{label}_energy_error_per_atom"])
                for record in valid_energy
            ],
            dtype=float,
        )
        model_force_records = [
            record
            for record in records
            if math.isfinite(
                float(record.get(f"_{label}_force_rmse", float("nan")))
            )
        ]
        summary[f"{label}_energy_mae_meV_atom"] = (
            float(np.mean(np.abs(model_energy_errors)) * 1000.0)
            if len(model_energy_errors)
            else float("nan")
        )
        summary[f"{label}_energy_rmse_meV_atom"] = (
            float(math.sqrt(np.mean(model_energy_errors**2)) * 1000.0)
            if len(model_energy_errors)
            else float("nan")
        )
        summary[f"{label}_force_rmse_eV_A"] = (
            float(
                math.sqrt(
                    np.mean(
                        [
                            float(record[f"_{label}_force_rmse"]) ** 2
                            for record in model_force_records
                        ]
                    )
                )
            )
            if model_force_records
            else float("nan")
        )
    return summary


def public_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.12f}"
    return value


def build_learning_curve(
    records: list[dict[str, object]], labels: list[str]
) -> list[dict[str, object]]:
    series_names = sorted(
        {str(record["series"]) for record in records},
        key=natural_key,
    )
    rows: list[dict[str, object]] = []
    cumulative: list[dict[str, object]] = []
    for series in series_names:
        current = [record for record in records if record["series"] == series]
        cumulative.extend(current)
        for scope, subset in (("round", current), ("cumulative", cumulative.copy())):
            row: dict[str, object] = {"scope": scope, "series": series}
            row.update(metric_summary(subset, labels))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: public_value(row.get(key, "")) for key in fieldnames})


def model_file_signatures(models: list[Path]) -> list[dict[str, object]]:
    signatures = []
    for model in models:
        stat = model.stat()
        signatures.append(
            {
                "path": str(model),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signatures


def compact_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(records: list[dict[str, object]]) -> str:
    payload = [
        {
            "directory": record.get("directory", ""),
            "structure_file": record.get("structure_file", ""),
            "n_atoms": record.get("n_atoms", ""),
            "formula": record.get("formula", ""),
            "dft_reference_source": record.get("dft_reference_source", ""),
            "dft_energy_eV": public_value(record.get("dft_energy_eV", "")),
        }
        for record in records
        if record.get("status") == "ok"
    ]
    return compact_fingerprint(payload)


def append_history_csv(path: Path, row: dict[str, object]) -> None:
    """Append a history row while allowing later runs to add metric columns."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        if path.exists() and path.stat().st_size > 0:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                existing_rows = list(reader)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow(existing)
                writer.writerow(
                    {
                        key: public_value(row.get(key, ""))
                        for key in fieldnames
                    }
                )
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_history_jsonl(path: Path, payload: dict[str, object]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=True) + "\n")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def archive_history(
    *,
    history_dir: Path,
    run_id: str,
    timestamp_utc: str,
    note: str,
    root: Path,
    args: argparse.Namespace,
    model_signatures: list[dict[str, object]],
    model_set_id: str,
    data_fingerprint: str,
    summary: dict[str, object],
    metrics_path: Path,
    learning_curve_path: Path,
    summary_path: Path,
) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    run_dir = history_dir / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    shutil.copy2(metrics_path, run_dir / "metrics.csv")
    shutil.copy2(learning_curve_path, run_dir / "learning_curve.csv")
    shutil.copy2(summary_path, run_dir / "summary.json")

    aggregate = dict(summary["aggregate"])
    series_rows = summary["per_series_and_cumulative"]
    series_names = sorted(
        {
            str(row["series"])
            for row in series_rows
            if row.get("scope") == "round"
        },
        key=natural_key,
    )
    history_row: dict[str, object] = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "note": note,
        "root": str(root),
        "recursive": bool(args.recursive),
        "indices": str(args.indices),
        "dataset_fingerprint": data_fingerprint,
        "model_set_id": model_set_id,
        "n_models": len(model_signatures),
        "n_series": len(series_names),
        "series": json.dumps(series_names),
        "models": json.dumps(model_signatures, sort_keys=True),
    }
    history_row.update(aggregate)
    append_history_csv(history_dir / "metrics_history.csv", history_row)

    event = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "note": note,
        "root": str(root),
        "command": [sys.executable, *sys.argv],
        "dataset_fingerprint": data_fingerprint,
        "model_set_id": model_set_id,
        "model_signatures": model_signatures,
        "archive_directory": str(run_dir),
        "summary": summary,
    }
    append_history_jsonl(history_dir / "runs.jsonl", event)
    return run_dir


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    root = Path(args.root).expanduser().resolve()
    models = resolve_models(
        args.model_path,
        args.calculator,
        minimum=1,
    )
    labels = model_labels(models)
    entries = iter_structure_dirs(
        root, parse_indices(args.indices), recursive=args.recursive
    )
    if not entries:
        raise FileNotFoundError(
            f"No matching numbered calculation directories found under {root}"
        )

    records: list[dict[str, object]] = []
    atoms_list = []
    dft_forces: list[np.ndarray | None] = []
    for relative, series, subdir, structure in entries:
        if structure is None:
            records.append(
                {
                    "directory": relative,
                    "series": series,
                    "structure_file": "",
                    "status": "missing_structure",
                }
            )
            continue
        atoms, dft_energy, forces, reference_source = read_dft_configuration(
            subdir, structure
        )
        records.append(
            {
                "directory": relative,
                "series": series,
                "structure_file": str(structure.relative_to(root)),
                "status": "ok",
                "n_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "dft_reference_source": reference_source,
                "dft_energy_eV": dft_energy,
                "_dft_energy": dft_energy,
            }
        )
        atoms_list.append(atoms)
        dft_forces.append(forces)

    valid_records = [record for record in records if record["status"] == "ok"]
    if not valid_records:
        raise ValueError("No readable structures were found.")

    n_configurations = len(atoms_list)
    n_models = len(models)
    energies = np.empty((n_configurations, n_models), dtype=float)
    forces_by_model: list[list[np.ndarray]] = [
        [np.empty((0, 3)) for _ in range(n_configurations)]
        for _ in range(n_models)
    ]
    elapsed_by_model = np.zeros(n_models, dtype=float)

    for model_index, (model, label) in enumerate(zip(models, labels)):
        backend = resolve_calculator_backend(model, args.calculator)
        calc = build_calculator(
            model,
            backend=backend,
            device=args.device,
            default_dtype=args.default_dtype,
            use_kokkos=args.use_kokkos,
        )
        print(
            f"Evaluating {label} ({model_index + 1}/{n_models}) "
            f"with {backend}: {model.name}"
        )
        tic_model = time.perf_counter()
        for config_index, atoms in enumerate(atoms_list):
            work = atoms.copy()
            work.calc = calc
            energies[config_index, model_index] = float(
                work.get_potential_energy()
            )
            forces_by_model[model_index][config_index] = np.asarray(
                work.get_forces(), dtype=float
            )
            print(
                f"  {valid_records[config_index]['directory']}: "
                f"E={energies[config_index, model_index]:.8f} eV"
            )
        elapsed_by_model[model_index] = time.perf_counter() - tic_model
        release_calculator(calc, args.device)

    traj_writer = None
    if args.traj_output:
        traj_path = Path(args.traj_output).expanduser().resolve()
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_writer = Trajectory(str(traj_path), "w")

    try:
        for config_index, (record, atoms, reference_forces) in enumerate(
            zip(valid_records, atoms_list, dft_forces)
        ):
            n_atoms = len(atoms)
            config_energies = energies[config_index]
            config_forces = np.stack(
                [
                    forces_by_model[model_index][config_index]
                    for model_index in range(n_models)
                ],
                axis=0,
            )
            energy_mean = float(np.mean(config_energies))
            energy_std = float(np.std(config_energies))
            force_mean = np.mean(config_forces, axis=0)
            force_std = np.std(config_forces, axis=0)
            force_std_rms = float(
                math.sqrt(np.mean(np.sum(force_std**2, axis=1)))
            )
            force_std_max = float(
                np.max(np.linalg.norm(force_std, axis=1))
            )
            dft_energy = float(record["_dft_energy"])
            ensemble_delta = (
                energy_mean - dft_energy
                if math.isfinite(dft_energy)
                else float("nan")
            )
            record.update(
                {
                    "committee_size": n_models,
                    "ensemble_energy_mean_eV": energy_mean,
                    "ensemble_energy_std_eV": energy_std,
                    "ensemble_energy_std_meV_atom": energy_std / n_atoms * 1000.0,
                    "ensemble_energy_error_eV": ensemble_delta,
                    "ensemble_energy_error_meV_atom": ensemble_delta / n_atoms * 1000.0,
                    "committee_force_std_rms_eV_A": force_std_rms,
                    "committee_force_std_max_eV_A": force_std_max,
                    "_ensemble_energy_error_per_atom": ensemble_delta / n_atoms,
                }
            )

            if reference_forces is not None:
                force_mae, force_rmse, force_max = force_error_stats(
                    force_mean, reference_forces
                )
            else:
                force_mae = force_rmse = force_max = float("nan")
            record.update(
                {
                    "ensemble_force_mae_eV_A": force_mae,
                    "ensemble_force_rmse_eV_A": force_rmse,
                    "ensemble_force_error_max_eV_A": force_max,
                    "_ensemble_force_mae": force_mae,
                    "_ensemble_force_rmse": force_rmse,
                }
            )

            for model_index, label in enumerate(labels):
                model_energy = float(config_energies[model_index])
                model_delta = (
                    model_energy - dft_energy
                    if math.isfinite(dft_energy)
                    else float("nan")
                )
                if reference_forces is not None:
                    _, model_force_rmse, _ = force_error_stats(
                        config_forces[model_index], reference_forces
                    )
                else:
                    model_force_rmse = float("nan")
                record.update(
                    {
                        f"{label}_energy_eV": model_energy,
                        f"{label}_energy_error_meV_atom": model_delta / n_atoms * 1000.0,
                        f"{label}_force_rmse_eV_A": model_force_rmse,
                        f"_{label}_energy_error_per_atom": model_delta / n_atoms,
                        f"_{label}_force_rmse": model_force_rmse,
                    }
                )

            output_atoms = atoms.copy()
            output_atoms.info.update(
                {
                    "source_directory": str(record["directory"]),
                    "committee_models": json.dumps([str(model) for model in models]),
                    "committee_energy_std_meV_atom": energy_std / n_atoms * 1000.0,
                    "committee_force_std_rms_eV_A": force_std_rms,
                }
            )
            if math.isfinite(dft_energy):
                output_atoms.info["energy_dft"] = dft_energy
            if reference_forces is not None:
                output_atoms.arrays["forces_dft"] = reference_forces
            output_atoms.arrays["committee_force_std"] = force_std
            output_atoms.calc = SinglePointCalculator(
                output_atoms, energy=energy_mean, forces=force_mean
            )
            if traj_writer is not None:
                traj_writer.write(output_atoms)
            if args.write_xyz:
                subdir = root / str(record["directory"])
                write(str(subdir / "mace_committee.xyz"), output_atoms)
    finally:
        if traj_writer is not None:
            traj_writer.close()

    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "directory",
        "series",
        "structure_file",
        "status",
        "n_atoms",
        "formula",
        "dft_reference_source",
        "dft_energy_eV",
        "committee_size",
        "ensemble_energy_mean_eV",
        "ensemble_energy_std_eV",
        "ensemble_energy_std_meV_atom",
        "ensemble_energy_error_eV",
        "ensemble_energy_error_meV_atom",
        "committee_force_std_rms_eV_A",
        "committee_force_std_max_eV_A",
        "ensemble_force_mae_eV_A",
        "ensemble_force_rmse_eV_A",
        "ensemble_force_error_max_eV_A",
    ]
    for label in labels:
        all_fields.extend(
            [
                f"{label}_energy_eV",
                f"{label}_energy_error_meV_atom",
                f"{label}_force_rmse_eV_A",
            ]
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for record in public_records:
            writer.writerow(
                {key: public_value(record.get(key, "")) for key in all_fields}
            )

    learning_rows = build_learning_curve(valid_records, labels)
    learning_curve_path = Path(args.learning_curve).expanduser().resolve()
    write_csv(learning_curve_path, learning_rows)
    timestamp = datetime.now(timezone.utc)
    timestamp_utc = timestamp.isoformat(timespec="microseconds")
    run_id = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    if args.history_note:
        note_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.history_note).strip("_")
        if note_slug:
            run_id = f"{run_id}_{note_slug[:48]}"
    model_signatures = model_file_signatures(models)
    model_set_id = compact_fingerprint(model_signatures)
    data_fingerprint = dataset_fingerprint(public_records)
    summary = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "history_note": args.history_note,
        "root": str(root),
        "models": [str(model) for model in models],
        "model_backends": [
            resolve_calculator_backend(model, args.calculator)
            for model in models
        ],
        "model_signatures": model_signatures,
        "model_set_id": model_set_id,
        "dataset_fingerprint": data_fingerprint,
        "aggregate": metric_summary(valid_records, labels),
        "per_series_and_cumulative": learning_rows,
        "model_elapsed_s": {
            label: float(elapsed)
            for label, elapsed in zip(labels, elapsed_by_model)
        },
    }
    summary_path = Path(args.summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    history_run_dir = None
    if not args.no_history:
        history_run_dir = archive_history(
            history_dir=Path(args.history_dir).expanduser().resolve(),
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            note=args.history_note,
            root=root,
            args=args,
            model_signatures=model_signatures,
            model_set_id=model_set_id,
            data_fingerprint=data_fingerprint,
            summary=summary,
            metrics_path=output,
            learning_curve_path=learning_curve_path,
            summary_path=summary_path,
        )
    print(f"Per-configuration metrics: {output}")
    print(f"Learning curve: {learning_curve_path}")
    print(f"Summary: {summary_path}")
    if history_run_dir is not None:
        print(f"History snapshot: {history_run_dir}")
    return public_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcmc-active-learning-evaluate",
        description=(
            "Evaluate a MACE/Symmetrix committee against VASP energies and forces "
            "in numbered active-learning directories."
        )
    )
    parser.add_argument(
        "--root",
        default=Path.cwd(),
        help="Calculation root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--indices",
        default="000-007",
        help="Directory indices/ranges, e.g. '000-007' or '0,2,5-7'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find numbered directories below all active-learning series under root.",
    )
    add_model_arguments(parser, default_device="cuda")
    parser.add_argument(
        "--output", default="mace_committee_metrics.csv"
    )
    parser.add_argument(
        "--summary", default="mace_committee_summary.json"
    )
    parser.add_argument(
        "--learning-curve", default="mace_learning_curve.csv"
    )
    parser.add_argument(
        "--history-dir",
        default="mace_history",
        help=(
            "Persistent history directory containing metrics_history.csv, "
            "runs.jsonl, and timestamped snapshots."
        ),
    )
    parser.add_argument(
        "--history-note",
        default="",
        help="Optional label for this model/data iteration.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not append or archive history for this invocation.",
    )
    parser.add_argument(
        "--traj-output", default="mace_committee.traj"
    )
    parser.add_argument("--write-xyz", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.traj_output == "":
        args.traj_output = None
    evaluate(args)


if __name__ == "__main__":
    main()
