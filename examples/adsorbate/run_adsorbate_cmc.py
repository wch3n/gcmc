import argparse
from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read

from gcmc import AdsorbateCMC


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


def build_calculator(args):
    if args.calculator == "mace":
        if not args.model:
            raise ValueError("--model is required when --calculator=mace.")
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=[args.model], device=args.device)

    from ase.calculators.lj import LennardJones

    return LennardJones(rc=args.lj_cutoff)


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


def main() -> None:
    default_snapshot = Path(__file__).resolve().with_name("POSCAR")

    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-loading NVT AdsorbateCMC on a MXene snapshot. "
            "By default this uses the shipped MXene snapshot in this directory, but "
            "--snapshot can point to any POSCAR or trajectory frame."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=default_snapshot)
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index when --snapshot is a trajectory-like file.",
    )
    parser.add_argument(
        "--calculator",
        choices=("mace", "lj"),
        default="mace",
        help="Use MACE for real runs or Lennard-Jones for a quick smoke test.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lj-cutoff", type=float, default=6.0)

    parser.add_argument(
        "--adsorbate",
        choices=("H", "OH", "OOH"),
        default="OH",
        help="Preset rigid adsorbate template.",
    )
    parser.add_argument("--temperature", type=float, default=600.0)
    parser.add_argument("--coverage", type=float, default=0.25)
    parser.add_argument("--site-type", choices=("atop", "fcc", "hcp"), default="atop")
    parser.add_argument("--surface-element", type=str, default="Ti")
    parser.add_argument(
        "--substrate-elements",
        type=str,
        default="Ti C",
        help="Whitespace or comma separated substrate elements.",
    )
    parser.add_argument(
        "--functional-elements",
        type=str,
        default=None,
        help="Optional whitespace/comma separated functional elements. Defaults to auto-detect from the clean snapshot.",
    )
    parser.add_argument(
        "--fix-below-z",
        type=float,
        default=None,
        help="Fix atoms with z < this value during optional relaxation.",
    )

    parser.add_argument(
        "--move-mode",
        choices=("displacement", "site_hop", "reorientation", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--site-hop-prob", type=float, default=0.35)
    parser.add_argument("--reorientation-prob", type=float, default=0.35)
    parser.add_argument("--rotation-max-angle-deg", type=float, default=25.0)
    parser.add_argument("--displacement-sigma", type=float, default=0.6)
    parser.add_argument("--max-displacement-trials", type=int, default=20)
    parser.add_argument("--max-reorientation-trials", type=int, default=20)
    parser.add_argument("--xy-tol", type=float, default=0.9)
    parser.add_argument("--support-xy-tol", type=float, default=1.5)
    parser.add_argument("--vertical-offset", type=float, default=1.8)

    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--relax-steps", type=int, default=20)
    parser.add_argument("--fmax", type=float, default=0.05)

    parser.add_argument("--nsweeps", type=int, default=200)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=2)
    parser.add_argument("--equilibration", type=int, default=40)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--output-prefix", type=str, default="adsorbate_cmc")

    args = parser.parse_args()

    atoms = load_snapshot(args.snapshot, args.frame)
    maybe_fix_bottom(atoms, args.fix_below_z)
    calculator = build_calculator(args)

    substrate_elements = parse_symbols(args.substrate_elements)
    if args.functional_elements:
        functional_elements = parse_symbols(args.functional_elements)
    else:
        functional_elements = tuple(
            sorted(
                set(atoms.get_chemical_symbols()) - set(substrate_elements)
            )
        )

    adsorbate_template, adsorbate_anchor_index = build_adsorbate_template(
        args.adsorbate
    )

    cmc = AdsorbateCMC.from_clean_surface(
        atoms=atoms,
        calculator=calculator,
        T=args.temperature,
        adsorbate=adsorbate_template,
        adsorbate_anchor_index=adsorbate_anchor_index,
        substrate_elements=substrate_elements,
        functional_elements=functional_elements,
        top_layer_element=args.surface_element,
        coverage=args.coverage,
        site_type=args.site_type,
        move_mode=args.move_mode,
        site_hop_prob=args.site_hop_prob,
        reorientation_prob=args.reorientation_prob,
        rotation_max_angle_deg=args.rotation_max_angle_deg,
        displacement_sigma=args.displacement_sigma,
        max_displacement_trials=args.max_displacement_trials,
        max_reorientation_trials=args.max_reorientation_trials,
        xy_tol=args.xy_tol,
        support_xy_tol=args.support_xy_tol,
        vertical_offset=args.vertical_offset,
        relax=args.relax,
        relax_steps=args.relax_steps,
        fmax=args.fmax,
        traj_file=f"{args.output_prefix}.traj",
        thermo_file=f"{args.output_prefix}.dat",
        checkpoint_file=f"{args.output_prefix}.pkl",
        initial_traj_file=f"{args.output_prefix}_initial.traj",
        seed=args.seed,
    )

    stats = cmc.run(
        nsweeps=args.nsweeps,
        traj_file=f"{args.output_prefix}.traj",
        interval=args.interval,
        sample_interval=args.sample_interval,
        equilibration=args.equilibration,
    )

    print("Final AdsorbateCMC stats:", stats)


if __name__ == "__main__":
    main()
