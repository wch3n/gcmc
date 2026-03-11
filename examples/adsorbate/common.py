from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read


def parse_symbols(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.replace(",", " ").split() if token)


def build_adsorbate_template(name: str) -> tuple[Atoms, int]:
    key = name.upper()
    if key == "H":
        return Atoms("H", positions=[(0.0, 0.0, 0.0)]), 0
    if key == "OH":
        return Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]), 0
    if key == "OOH":
        ooh = Atoms(
            "OOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.46),
                (0.79, 0.0, 2.01),
            ],
        )
        return ooh, 0
    raise ValueError(f"Unsupported preset adsorbate '{name}'. Use H, OH, or OOH.")


def get_calculator_class_and_kwargs(args):
    if args.calculator == "mace":
        if not args.model:
            raise ValueError("--model is required when --calculator=mace.")
        from mace.calculators import MACECalculator

        return MACECalculator, {"model_paths": [args.model], "device": args.device}

    from ase.calculators.lj import LennardJones

    return LennardJones, {"rc": args.lj_cutoff}


def build_calculator(args):
    calculator_class, calc_kwargs = get_calculator_class_and_kwargs(args)
    return calculator_class(**calc_kwargs)


def load_snapshot(path: Path, frame: int) -> Atoms:
    suffix = path.suffix.lower()
    index = frame if suffix in {".traj", ".db", ".extxyz", ".xyz"} else 0
    return read(path, index=index)


def maybe_fix_bottom(atoms: Atoms, z_cut: float | None) -> None:
    if z_cut is None:
        return
    fixed_indices = [atom.index for atom in atoms if atom.position[2] < z_cut]
    if fixed_indices:
        atoms.set_constraint(FixAtoms(indices=fixed_indices))


def infer_functional_elements(
    atoms: Atoms,
    substrate_elements: tuple[str, ...],
    explicit: str | None = None,
) -> tuple[str, ...]:
    if explicit:
        return parse_symbols(explicit)
    return tuple(sorted(set(atoms.get_chemical_symbols()) - set(substrate_elements)))
