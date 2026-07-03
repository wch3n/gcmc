#!/usr/bin/env python3
"""Check binary alloy trajectory energies against endpoint tie lines.

This script is intended to run on a cluster/GPU node with the same MLIP
environment used for alloy CMC/PT.  It compares sampled alloy energies against
two endpoint structures and reports the energy above/below the binary tie line.

For MXenes such as (Ti,Zr)2CO2, use ``--formula-count-element C`` so energies
are normalized per M2CO2 formula unit.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import yaml
from ase import Atoms
from ase.build import make_supercell
from ase.io import iread, read
from ase.optimize import BFGS

from gcmc.workflows import build_adsorbate_gcmc_calculator


_KB_EV_PER_K = 8.617333262145e-5


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _calculator_from_config(config: dict[str, object]) -> object:
    calc_cfg = dict(_section(config, "calculator"))
    if not calc_cfg:
        raise ValueError("Config does not contain a calculator block.")
    task = {}
    if calc_cfg.get("device") is not None:
        task["device"] = calc_cfg["device"]
    return build_adsorbate_gcmc_calculator(SimpleNamespace(**calc_cfg), task)


def _calculator_metadata(config: dict[str, object]) -> dict[str, object]:
    calc_cfg = dict(_section(config, "calculator"))
    calculator = calc_cfg.get("calculator", calc_cfg.get("type", ""))
    model = calc_cfg.get("model", calc_cfg.get("model_file", ""))
    device = calc_cfg.get("device", "")
    return {
        "calculator": calculator,
        "model": model,
        "device": device,
    }


def _as_supercell_matrix(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 1 and arr.size == 3:
        return np.diag(arr)
    if arr.ndim == 1 and arr.size == 9:
        return arr.reshape(3, 3)
    raise ValueError("supercell_matrix must be a 3x3 matrix or 3/9 integers.")


def _formula_count(atoms: Atoms, element: str | None, metals: Sequence[str]) -> float:
    symbols = atoms.get_chemical_symbols()
    if element:
        count = symbols.count(element)
        if count <= 0:
            raise ValueError(f"No {element!r} atoms found for formula normalization.")
        return float(count)
    metal_count = sum(symbols.count(element) for element in metals)
    if metal_count <= 0:
        raise ValueError("No alloy metal atoms found for formula normalization.")
    return float(metal_count) / 2.0


def _composition_x(atoms: Atoms, element_a: str, element_b: str) -> float:
    symbols = atoms.get_chemical_symbols()
    na = symbols.count(element_a)
    nb = symbols.count(element_b)
    if na + nb <= 0:
        raise ValueError(f"No {element_a}/{element_b} atoms found in alloy structure.")
    return na / (na + nb)


def _maybe_relax(
    atoms: Atoms,
    calc: object,
    *,
    relax: bool,
    fmax: float,
    steps: int,
) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = calc
    if relax:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=steps)
    return atoms


def _energy(atoms: Atoms, calc: object) -> float:
    atoms = atoms.copy()
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def _replace_alloy_sites(
    atoms: Atoms,
    *,
    source_site_element: str | None,
    alloy_elements: Sequence[str],
    target_element: str,
) -> Atoms:
    atoms = atoms.copy()
    symbols = atoms.get_chemical_symbols()
    replace = set(alloy_elements)
    if source_site_element:
        replace.add(source_site_element)
    for idx, symbol in enumerate(symbols):
        if symbol in replace:
            symbols[idx] = target_element
    atoms.set_chemical_symbols(symbols)
    return atoms


def _build_coherent_endpoint(
    config: dict[str, object],
    config_path: Path,
    *,
    alloy_elements: Sequence[str],
    target_element: str,
) -> Atoms:
    system = _section(config, "system")
    snapshot = system.get("snapshot")
    if snapshot is None:
        raise ValueError("Config system.snapshot is needed for coherent endpoint construction.")
    primitive = read(_resolve_path(str(snapshot), base_dir=config_path.parent))
    matrix = _as_supercell_matrix(system.get("supercell_matrix", np.eye(3, dtype=int)))
    atoms = make_supercell(primitive, matrix)
    return _replace_alloy_sites(
        atoms,
        source_site_element=str(system.get("site_element")) if system.get("site_element") else None,
        alloy_elements=alloy_elements,
        target_element=target_element,
    )


def _parse_dat(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values: list[float] = []
            for token in stripped.replace(",", " ").split():
                try:
                    values.append(float(token))
                except ValueError:
                    pass
            if len(values) >= 2:
                rows.append({"cycle": values[0], "energy_eV": values[1]})
    return rows


def _selected_dat_rows(
    rows: Sequence[dict[str, float]],
    *,
    start: int,
    stop: int | None,
    step: int,
) -> list[dict[str, float]]:
    return list(rows[start:stop:step])


def _selected_traj_energies(
    traj: Path,
    calc: object,
    *,
    start: int,
    stop: int | None,
    step: int,
    quench: bool,
    fmax: float,
    steps: int,
    max_frames: int | None,
    progress_interval: int,
) -> list[dict[str, float]]:
    index = f"{start}:{'' if stop is None else stop}:{step}"
    rows: list[dict[str, float]] = []
    for local_idx, atoms in enumerate(iread(str(traj), index=index)):
        if max_frames is not None and len(rows) >= max_frames:
            break
        frame = start + local_idx * step
        sample_id = len(rows) + 1
        if progress_interval > 0 and (sample_id == 1 or sample_id % progress_interval == 0):
            action = "quenching" if quench else "evaluating"
            limit = f"/{max_frames}" if max_frames is not None else ""
            print(f"[alloy] {action} sample {sample_id}{limit} from frame {frame}", flush=True)
        initial_energy = _energy(atoms, calc) if quench else None
        relaxed = _maybe_relax(atoms, calc, relax=quench, fmax=fmax, steps=steps)
        energy = _energy(relaxed, calc)
        if progress_interval > 0 and (sample_id == 1 or sample_id % progress_interval == 0):
            if initial_energy is None:
                print(f"[alloy] sample {sample_id} energy = {energy:.8f} eV", flush=True)
            else:
                print(
                    f"[alloy] sample {sample_id} quenched energy = {energy:.8f} eV "
                    f"(delta {energy - initial_energy:+.8f} eV)",
                    flush=True,
                )
        row = {
            "cycle": float(frame),
            "frame": int(frame),
            "energy_eV": energy,
            "quenched": bool(quench),
            "n_atoms": len(relaxed),
        }
        if initial_energy is not None:
            row["initial_energy_eV"] = initial_energy
            row["quench_delta_eV"] = energy - initial_energy
        rows.append(row)
    return rows


def _ideal_config_free_energy_per_formula(
    *,
    x: float,
    temperature_K: float,
    metals_per_formula: float,
) -> float:
    if temperature_K <= 0.0 or x <= 0.0 or x >= 1.0:
        return 0.0
    s_term = x * math.log(x) + (1.0 - x) * math.log(1.0 - x)
    return metals_per_formula * _KB_EV_PER_K * temperature_K * s_term


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_pyplot():
    mpl_config = os.environ.get("MPLCONFIGDIR")
    if not mpl_config or not os.access(Path(mpl_config).expanduser(), os.W_OK):
        cache_dir = Path(tempfile.gettempdir()) / f"gcmc-matplotlib-{os.getuid()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return None, exc
    return plt, None


def _write_plot(
    path: Path,
    *,
    rows: Sequence[dict[str, object]],
    reference_kind: str,
    elements: Sequence[str],
    plot_format: str,
) -> Path | None:
    plt, error = _load_pyplot()
    path.mkdir(parents=True, exist_ok=True)
    if error is not None:
        (path / "plotting_error.txt").write_text(str(error) + "\n")
        return None
    subset = [row for row in rows if row["reference_kind"] == reference_kind]
    if not subset:
        return None
    cycles = np.asarray([float(row["cycle"]) for row in subset], dtype=float)
    values = np.asarray([float(row["delta_e_mix_meV_per_metal"]) for row in subset], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(cycles, values, marker=".", linestyle="none", alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("sample cycle/frame")
    ax.set_ylabel(f"energy above {elements[0]}/{elements[1]} tie line (meV/metal)")
    ax.grid(alpha=0.25)
    out = path / f"hull_distance_{reference_kind}.{plot_format}"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check binary alloy trajectory energies against endpoint convex-hull tie lines.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Alloy workflow config.yaml containing the MLIP calculator.")
    parser.add_argument("--traj", type=Path, required=True, help="Alloy trajectory used for composition/formula counts.")
    parser.add_argument("--dat", type=Path, default=None, help="Optional alloy .dat energy trace. Used unless --recompute-alloy is set.")
    parser.add_argument("--elements", nargs=2, required=True, metavar=("A", "B"), help="Binary alloy elements, e.g. Ti Zr.")
    parser.add_argument("--endmember", nargs=2, action="append", metavar=("ELEMENT", "PATH"), required=True, help="Endpoint structure, e.g. --endmember Ti POSCAR.Ti")
    parser.add_argument("--formula-count-element", default="C", help="Element count used as formula-unit count; use '' to fall back to metal_count/2.")
    parser.add_argument("--metals-per-formula", type=float, default=2.0, help="Number of alloy metal sites per normalized formula unit.")
    parser.add_argument("--temperature-K", type=float, default=300.0, help="Temperature for ideal configurational free-energy correction.")
    parser.add_argument("--start", type=int, default=0, help="First .dat sample or trajectory frame.")
    parser.add_argument("--stop", type=int, default=None, help="Stop .dat sample or trajectory frame.")
    parser.add_argument("--step", type=int, default=1, help="Stride for .dat samples or trajectory frames.")
    parser.add_argument("--recompute-alloy", action="store_true", help="Recompute alloy frame energies with the MLIP instead of using --dat.")
    parser.add_argument("--quench-alloy", action="store_true", help="Relax selected alloy trajectory frames before hull comparison. Implies --recompute-alloy.")
    parser.add_argument("--alloy-fmax", type=float, default=0.03, help="Alloy-frame quench fmax in eV/A.")
    parser.add_argument("--alloy-relax-steps", type=int, default=300, help="Maximum relaxation steps for each selected alloy frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum selected alloy trajectory frames to evaluate.")
    parser.add_argument("--progress-interval", type=int, default=1, help="Print progress every N selected alloy frames; use 0 to disable.")
    parser.add_argument("--relax-endpoints", action="store_true", help="Relax endpoint atomic positions before evaluating endpoint energies.")
    parser.add_argument("--endpoint-fmax", type=float, default=0.03, help="Endpoint relaxation fmax in eV/A.")
    parser.add_argument("--endpoint-relax-steps", type=int, default=300, help="Maximum endpoint relaxation steps.")
    parser.add_argument("--coherent-endpoints", action="store_true", help="Also evaluate pure endpoints in the alloy mean-cell supercell from config.")
    parser.add_argument("--no-free-endpoints", action="store_true", help="Skip endpoint structures supplied by --endmember.")
    parser.add_argument("--out-dir", type=Path, default=Path("binary_hull_check"), help="Output directory.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Plot format.")
    args = parser.parse_args()

    config = _load_config(args.config)
    calc_meta = _calculator_metadata(config)
    print(
        "[calculator] "
        f"type={calc_meta.get('calculator') or 'unknown'} "
        f"model={calc_meta.get('model') or 'not set'} "
        f"device={calc_meta.get('device') or 'default'}",
        flush=True,
    )
    calc = _calculator_from_config(config)
    element_a, element_b = args.elements
    endpoint_paths = {element: Path(path).expanduser().resolve() for element, path in args.endmember}
    missing = [element for element in args.elements if element not in endpoint_paths]
    if missing:
        raise ValueError(f"Missing --endmember entries for: {', '.join(missing)}")

    formula_element = args.formula_count_element or None
    first_alloy = read(args.traj, index=0)
    alloy_formula_count = _formula_count(first_alloy, formula_element, args.elements)
    x_a = _composition_x(first_alloy, element_a, element_b)

    endpoint_rows: list[dict[str, object]] = []
    endpoint_energy: dict[str, dict[str, float]] = {}

    if not args.no_free_endpoints:
        endpoint_energy["free"] = {}
        for element in args.elements:
            print(f"[endpoint/free] evaluating {element}: {endpoint_paths[element]}", flush=True)
            atoms = read(endpoint_paths[element])
            atoms = _maybe_relax(
                atoms,
                calc,
                relax=args.relax_endpoints,
                fmax=args.endpoint_fmax,
                steps=args.endpoint_relax_steps,
            )
            energy = _energy(atoms, calc)
            n_formula = _formula_count(atoms, formula_element, args.elements)
            per_formula = energy / n_formula
            print(
                f"[endpoint/free] {element} energy = {energy:.8f} eV "
                f"({per_formula:.8f} eV/formula)",
                flush=True,
            )
            endpoint_energy["free"][element] = per_formula
            endpoint_rows.append(
                {
                    "reference_kind": "free",
                    "element": element,
                    "structure": str(endpoint_paths[element]),
                    "relaxed": bool(args.relax_endpoints),
                    "n_atoms": len(atoms),
                    "n_formula": n_formula,
                    "energy_eV": energy,
                    "energy_eV_per_formula": per_formula,
                }
            )

    if args.coherent_endpoints:
        endpoint_energy["coherent"] = {}
        for element in args.elements:
            print(f"[endpoint/coherent] evaluating {element} in alloy mean-cell supercell", flush=True)
            atoms = _build_coherent_endpoint(
                config,
                args.config.resolve(),
                alloy_elements=args.elements,
                target_element=element,
            )
            atoms = _maybe_relax(
                atoms,
                calc,
                relax=args.relax_endpoints,
                fmax=args.endpoint_fmax,
                steps=args.endpoint_relax_steps,
            )
            energy = _energy(atoms, calc)
            n_formula = _formula_count(atoms, formula_element, args.elements)
            per_formula = energy / n_formula
            print(
                f"[endpoint/coherent] {element} energy = {energy:.8f} eV "
                f"({per_formula:.8f} eV/formula)",
                flush=True,
            )
            endpoint_energy["coherent"][element] = per_formula
            endpoint_rows.append(
                {
                    "reference_kind": "coherent",
                    "element": element,
                    "structure": "config mean-cell supercell",
                    "relaxed": bool(args.relax_endpoints),
                    "n_atoms": len(atoms),
                    "n_formula": n_formula,
                    "energy_eV": energy,
                    "energy_eV_per_formula": per_formula,
                }
            )

    if not endpoint_energy:
        raise ValueError("No endpoint reference was evaluated.")

    if args.quench_alloy:
        args.recompute_alloy = True

    if args.recompute_alloy:
        alloy_rows = _selected_traj_energies(
            args.traj,
            calc,
            start=args.start,
            stop=args.stop,
            step=args.step,
            quench=args.quench_alloy,
            fmax=args.alloy_fmax,
            steps=args.alloy_relax_steps,
            max_frames=args.max_frames,
            progress_interval=args.progress_interval,
        )
        alloy_energy_source = "quenched_traj" if args.quench_alloy else "recomputed_traj"
    else:
        raw_rows = _parse_dat(args.dat)
        if not raw_rows:
            raise ValueError("No .dat energies found. Pass --dat or use --recompute-alloy.")
        alloy_rows = _selected_dat_rows(raw_rows, start=args.start, stop=args.stop, step=args.step)
        if args.max_frames is not None:
            alloy_rows = alloy_rows[: args.max_frames]
        alloy_energy_source = str(args.dat)
    if not alloy_rows:
        raise ValueError("No alloy samples selected. Check --start/--stop/--step/--max-frames.")

    entropy_eV_per_formula = _ideal_config_free_energy_per_formula(
        x=x_a,
        temperature_K=args.temperature_K,
        metals_per_formula=args.metals_per_formula,
    )
    sample_rows: list[dict[str, object]] = []
    for sample_id, row in enumerate(alloy_rows):
        alloy_e_per_formula = float(row["energy_eV"]) / alloy_formula_count
        for reference_kind, refs in endpoint_energy.items():
            tie_line = x_a * refs[element_a] + (1.0 - x_a) * refs[element_b]
            delta = alloy_e_per_formula - tie_line
            free_delta = delta + entropy_eV_per_formula
            sample_rows.append(
                {
                    "sample": sample_id,
                    "cycle": row["cycle"],
                    "reference_kind": reference_kind,
                    "alloy_energy_source": alloy_energy_source,
                    f"x_{element_a}": x_a,
                    f"x_{element_b}": 1.0 - x_a,
                    "alloy_energy_eV": row["energy_eV"],
                    "alloy_initial_energy_eV": row.get("initial_energy_eV", ""),
                    "alloy_quench_delta_eV": row.get("quench_delta_eV", ""),
                    "alloy_quenched": row.get("quenched", False),
                    "alloy_energy_eV_per_formula": alloy_e_per_formula,
                    "tie_line_energy_eV_per_formula": tie_line,
                    "delta_e_mix_eV_per_formula": delta,
                    "delta_e_mix_meV_per_metal": 1000.0 * delta / args.metals_per_formula,
                    "ideal_config_free_energy_eV_per_formula": entropy_eV_per_formula,
                    "delta_g_ideal_eV_per_formula": free_delta,
                    "delta_g_ideal_meV_per_metal": 1000.0 * free_delta / args.metals_per_formula,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for reference_kind in endpoint_energy:
        subset = [row for row in sample_rows if row["reference_kind"] == reference_kind]
        values = np.asarray([float(row["delta_e_mix_meV_per_metal"]) for row in subset], dtype=float)
        g_values = np.asarray([float(row["delta_g_ideal_meV_per_metal"]) for row in subset], dtype=float)
        summary_rows.append(
            {
                "reference_kind": reference_kind,
                "n_samples": len(subset),
                f"x_{element_a}": x_a,
                "temperature_K": args.temperature_K,
                "delta_e_min_meV_per_metal": float(np.min(values)),
                "delta_e_mean_meV_per_metal": float(np.mean(values)),
                "delta_e_max_meV_per_metal": float(np.max(values)),
                "delta_g_ideal_min_meV_per_metal": float(np.min(g_values)),
                "delta_g_ideal_mean_meV_per_metal": float(np.mean(g_values)),
                "delta_g_ideal_max_meV_per_metal": float(np.max(g_values)),
                "ideal_config_free_energy_meV_per_metal": 1000.0 * entropy_eV_per_formula / args.metals_per_formula,
            }
        )

    out_dir = args.out_dir
    _write_csv(out_dir / "endpoint_energies.csv", endpoint_rows)
    _write_csv(out_dir / "alloy_hull_samples.csv", sample_rows)
    _write_csv(out_dir / "hull_summary.csv", summary_rows)
    plot_paths = [
        _write_plot(
            out_dir / "plots",
            rows=sample_rows,
            reference_kind=kind,
            elements=args.elements,
            plot_format=args.plot_format,
        )
        for kind in endpoint_energy
    ]
    plot_paths = [path for path in plot_paths if path is not None]

    lines = [
        "# Binary Alloy Hull Check",
        "",
        f"Alloy trajectory: `{args.traj}`",
        f"Energy source: `{alloy_energy_source}`",
        f"Alloy quench: `{bool(args.quench_alloy)}`",
        f"Calculator: `{calc_meta.get('calculator') or 'unknown'}`",
        f"Model: `{calc_meta.get('model') or 'not set'}`",
        f"Device: `{calc_meta.get('device') or 'default'}`",
        f"Composition: x_{element_a} = {x_a:.6f}",
        f"Formula count: {alloy_formula_count:g}",
        f"Ideal configurational term at {args.temperature_K:g} K: "
        f"{1000.0 * entropy_eV_per_formula / args.metals_per_formula:.3f} meV/metal",
        "",
        "## Summary",
    ]
    for row in summary_rows:
        lines.append(
            "- {reference_kind}: DeltaE mean/min/max = "
            "{delta_e_mean_meV_per_metal:.3f} / {delta_e_min_meV_per_metal:.3f} / "
            "{delta_e_max_meV_per_metal:.3f} meV/metal; "
            "DeltaG_ideal mean = {delta_g_ideal_mean_meV_per_metal:.3f} meV/metal".format(**row)
        )
    if plot_paths:
        lines.extend(["", "## Plots", *[f"- `{path}`" for path in plot_paths]])
    (out_dir / "hull_report.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote binary hull check to: {out_dir}")
    for row in summary_rows:
        print(
            "{reference_kind}: DeltaE mean={delta_e_mean_meV_per_metal:.3f} "
            "min={delta_e_min_meV_per_metal:.3f} max={delta_e_max_meV_per_metal:.3f} "
            "meV/metal; DeltaG_ideal mean={delta_g_ideal_mean_meV_per_metal:.3f} meV/metal".format(**row)
        )


if __name__ == "__main__":
    main()
