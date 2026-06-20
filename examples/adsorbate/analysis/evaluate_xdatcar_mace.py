#!/usr/bin/env python3
"""Evaluate selected XDATCAR frames with a MACE calculator."""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.io import iread
from ase.io.trajectory import Trajectory


_OSZICAR_FINAL_RE = re.compile(
    r"^\s*(?P<step>\d+)\s+F=\s*(?P<F>[-+0-9.Ee]+)"
    r"\s+E0=\s*(?P<E0>[-+0-9.Ee]+)"
)


def _parse_indices(text: str | None) -> set[int] | None:
    if text is None or not text.strip():
        return None
    indices: set[int] = set()
    for token in text.replace(",", " ").split():
        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid index range token: {token!r}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError("Index range step must be positive.")
            indices.update(range(start, stop, step))
        else:
            indices.add(int(token))
    return indices


def _frame_is_selected(
    frame_index: int,
    *,
    indices: set[int] | None,
    start: int,
    stop: int | None,
    step: int,
    max_frames: int | None,
    selected_count: int,
) -> bool:
    if max_frames is not None and selected_count >= max_frames:
        return False
    if indices is not None:
        return frame_index in indices
    if frame_index < start:
        return False
    if stop is not None and frame_index >= stop:
        return False
    return (frame_index - start) % step == 0


def _iter_selected_frames(
    xdatcar: Path,
    *,
    indices: set[int] | None,
    start: int,
    stop: int | None,
    step: int,
    max_frames: int | None,
):
    selected_count = 0
    for frame_index, atoms in enumerate(iread(str(xdatcar), index=":")):
        if stop is not None and indices is None and frame_index >= stop:
            break
        if _frame_is_selected(
            frame_index,
            indices=indices,
            start=start,
            stop=stop,
            step=step,
            max_frames=max_frames,
            selected_count=selected_count,
        ):
            selected_count += 1
            yield frame_index, atoms
        if max_frames is not None and selected_count >= max_frames:
            break


def _read_oszicar_energies(path: Path, field: str) -> list[float]:
    values: list[float] = []
    field = field.upper()
    with path.open() as handle:
        for line in handle:
            match = _OSZICAR_FINAL_RE.match(line)
            if not match:
                continue
            values.append(float(match.group(field)))
    return values


def _reference_energies(
    xdatcar: Path,
    oszicar: Path | None,
    field: str,
    no_reference: bool,
) -> list[float]:
    if no_reference:
        return []
    if oszicar is None:
        candidate = xdatcar.with_name("OSZICAR")
        if not candidate.exists():
            return []
        oszicar = candidate
    if not oszicar.exists():
        raise FileNotFoundError(f"Reference OSZICAR not found: {oszicar}")
    return _read_oszicar_energies(oszicar, field)


def _build_mace_calculator(model: Path, device: str, default_dtype: str | None):
    from mace.calculators import MACECalculator

    kwargs = {
        "model_paths": [str(model)],
        "device": str(device),
    }
    if default_dtype:
        kwargs["default_dtype"] = str(default_dtype)
    return MACECalculator(**kwargs)


def _force_stats(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(np.asarray(forces, dtype=float), axis=1)
    return float(np.max(norms)), float(math.sqrt(np.mean(norms**2)))


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "frame_index",
        "n_atoms",
        "formula",
        "mace_energy_eV",
        "mace_energy_per_atom_eV",
        "force_max_eV_A",
        "force_rms_eV_A",
        "reference_energy_eV",
        "delta_mace_minus_reference_eV",
        "delta_mace_minus_reference_per_atom_eV",
        "elapsed_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> list[dict[str, object]]:
    xdatcar = Path(args.xdatcar).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    indices = _parse_indices(args.indices)
    reference = _reference_energies(
        xdatcar,
        Path(args.reference_oszicar).expanduser().resolve()
        if args.reference_oszicar
        else None,
        args.reference_field,
        args.no_reference,
    )
    calc = _build_mace_calculator(model, args.device, args.default_dtype)

    traj_writer = None
    if args.traj_output:
        traj_path = Path(args.traj_output).expanduser().resolve()
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_writer = Trajectory(str(traj_path), "w")

    rows: list[dict[str, object]] = []
    try:
        for selected_count, (frame_index, atoms) in enumerate(
            _iter_selected_frames(
                xdatcar,
                indices=indices,
                start=args.start,
                stop=args.stop,
                step=args.step,
                max_frames=args.max_frames,
            ),
            start=1,
        ):
            work = atoms.copy()
            work.calc = calc
            tic = time.perf_counter()
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), dtype=float)
            elapsed = float(time.perf_counter() - tic)
            force_max, force_rms = _force_stats(forces)
            n_atoms = len(work)
            reference_energy = (
                float(reference[frame_index])
                if 0 <= frame_index < len(reference)
                else float("nan")
            )
            delta = (
                energy - reference_energy
                if math.isfinite(reference_energy)
                else float("nan")
            )
            row = {
                "frame_index": frame_index,
                "n_atoms": n_atoms,
                "formula": work.get_chemical_formula(),
                "mace_energy_eV": energy,
                "mace_energy_per_atom_eV": energy / n_atoms,
                "force_max_eV_A": force_max,
                "force_rms_eV_A": force_rms,
                "reference_energy_eV": reference_energy,
                "delta_mace_minus_reference_eV": delta,
                "delta_mace_minus_reference_per_atom_eV": delta / n_atoms
                if math.isfinite(delta)
                else float("nan"),
                "elapsed_s": elapsed,
            }
            rows.append(row)

            if traj_writer is not None:
                work.info.update(row)
                work.arrays["mace_forces"] = forces
                traj_writer.write(work)

            if args.report_interval > 0 and selected_count % args.report_interval == 0:
                print(
                    f"frame {frame_index}: E_MACE={energy:.8f} eV, "
                    f"Fmax={force_max:.4f} eV/A, elapsed={elapsed:.2f}s"
                )
    finally:
        if traj_writer is not None:
            traj_writer.close()

    _write_rows(output, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read selected frames from a VASP XDATCAR and evaluate single-point "
            "energies/forces with a MACE model."
        )
    )
    parser.add_argument("--xdatcar", required=True, help="Path to VASP XDATCAR.")
    parser.add_argument("--model", required=True, help="Path to MACE model file.")
    parser.add_argument("--device", default="cuda", help="MACE device, e.g. cuda or cpu.")
    parser.add_argument(
        "--default-dtype",
        default=None,
        help="Optional MACE default dtype, e.g. float32 or float64.",
    )
    parser.add_argument("--start", type=int, default=0, help="First zero-based frame.")
    parser.add_argument("--stop", type=int, default=None, help="Exclusive stop frame.")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--indices",
        default=None,
        help="Explicit zero-based frames/ranges, e.g. '0,10,20:100:5'.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of selected frames to evaluate.",
    )
    parser.add_argument(
        "--reference-oszicar",
        default=None,
        help="Optional OSZICAR path. Defaults to OSZICAR next to XDATCAR if present.",
    )
    parser.add_argument(
        "--reference-field",
        choices=("F", "E0"),
        default="F",
        help="OSZICAR energy field used as reference.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Do not read OSZICAR reference energies.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--traj-output",
        default=None,
        help="Optional ASE trajectory with MACE energy metadata and force arrays.",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=1,
        help="Print progress every N evaluated frames; set 0 to silence.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    rows = evaluate(args)
    print(f"Wrote {len(rows)} rows to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
