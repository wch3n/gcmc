import argparse
from pathlib import Path

from gcmc import AdsorbateCMC, ReplicaExchange
from common import (
    build_adsorbate_template,
    build_calculator,
    get_calculator_class_and_kwargs,
    infer_functional_elements,
    load_snapshot,
    maybe_fix_bottom,
    parse_symbols,
)


def main() -> None:
    default_snapshot = Path(__file__).resolve().parents[1] / "alloy" / "POSCAR.Ti2CO2"

    parser = argparse.ArgumentParser(
        description=(
            "Run replica exchange with AdsorbateCMC on a fixed-loading MXene "
            "snapshot. The initial adsorbate configuration is built once on the "
            "driver and then exchanged across the temperature ladder."
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
        "--repeat",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(2, 2, 1),
        help="Repeat the input snapshot before building the initial adsorbate configuration.",
    )
    parser.add_argument(
        "--calculator",
        choices=("mace", "lj"),
        default="mace",
        help="Use MACE for production or Lennard-Jones for a quick smoke test.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lj-cutoff", type=float, default=6.0)

    parser.add_argument("--adsorbate", choices=("H", "OH", "OOH"), default="OH")
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

    parser.add_argument("--t-start", type=float, default=900.0)
    parser.add_argument("--t-end", type=float, default=300.0)
    parser.add_argument(
        "--t-step",
        type=float,
        default=100.0,
        help="Used for a uniform ladder when --n-replicas is not set.",
    )
    parser.add_argument(
        "--n-replicas",
        type=int,
        default=None,
        help="Set this to use an auto-generated ladder instead of --t-step.",
    )
    parser.add_argument(
        "--grid-space",
        choices=("temperature", "beta"),
        default="temperature",
        help="Baseline spacing when --n-replicas is used.",
    )

    parser.add_argument("--n-cycles", type=int, default=20)
    parser.add_argument("--equilibration-cycles", type=int, default=2)
    parser.add_argument("--swap-interval", type=int, default=20)
    parser.add_argument("--report-interval", type=int, default=5)
    parser.add_argument("--sampling-interval", type=int, default=2)
    parser.add_argument("--local-eq-fraction", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--swap-stride", type=int, default=1)
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--execution-backend",
        choices=("multiprocessing", "ray"),
        default="multiprocessing",
    )
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--output-prefix", type=str, default="adsorbate_pt")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    atoms = load_snapshot(args.snapshot, args.frame)
    if tuple(args.repeat) != (1, 1, 1):
        atoms = atoms.repeat(tuple(args.repeat))
    maybe_fix_bottom(atoms, args.fix_below_z)

    substrate_elements = parse_symbols(args.substrate_elements)
    functional_elements = infer_functional_elements(
        atoms, substrate_elements, args.functional_elements
    )
    adsorbate_template, adsorbate_anchor_index = build_adsorbate_template(
        args.adsorbate
    )

    calculator = build_calculator(args)
    seed_builder = AdsorbateCMC.from_clean_surface(
        atoms=atoms,
        calculator=calculator,
        T=args.t_start,
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
        traj_file=f"{args.output_prefix}_seed.traj",
        thermo_file=f"{args.output_prefix}_seed.dat",
        checkpoint_file=f"{args.output_prefix}_seed.pkl",
        initial_traj_file=f"{args.output_prefix}_initial.traj",
        seed=args.seed,
    )

    calculator_class, calc_kwargs = get_calculator_class_and_kwargs(args)
    pt_kwargs = dict(
        atoms_template=seed_builder.atoms.copy(),
        T_start=args.t_start,
        T_end=args.t_end,
        T_step=args.t_step,
        n_replicas=args.n_replicas,
        grid_space=args.grid_space,
        calculator_class=calculator_class,
        calc_kwargs=calc_kwargs,
        mc_class=AdsorbateCMC,
        mc_kwargs={
            "adsorbate": adsorbate_template,
            "adsorbate_anchor_index": adsorbate_anchor_index,
            "substrate_elements": substrate_elements,
            "functional_elements": functional_elements,
            "top_layer_element": args.surface_element,
            "coverage": args.coverage,
            "site_type": args.site_type,
            "move_mode": args.move_mode,
            "site_hop_prob": args.site_hop_prob,
            "reorientation_prob": args.reorientation_prob,
            "rotation_max_angle_deg": args.rotation_max_angle_deg,
            "displacement_sigma": args.displacement_sigma,
            "max_displacement_trials": args.max_displacement_trials,
            "max_reorientation_trials": args.max_reorientation_trials,
            "xy_tol": args.xy_tol,
            "support_xy_tol": args.support_xy_tol,
            "vertical_offset": args.vertical_offset,
            "relax": args.relax,
            "relax_steps": args.relax_steps,
            "fmax": args.fmax,
            "checkpoint_interval": 0,
        },
        n_gpus=args.n_gpus,
        workers_per_gpu=args.workers_per_gpu,
        swap_interval=args.swap_interval,
        report_interval=args.report_interval,
        sampling_interval=args.sampling_interval,
        local_eq_fraction=args.local_eq_fraction,
        checkpoint_interval=args.checkpoint_interval,
        swap_stride=args.swap_stride,
        execution_backend=args.execution_backend,
        results_file=f"{args.output_prefix}_results.csv",
        stats_file=f"{args.output_prefix}_stats.csv",
        checkpoint_file=f"{args.output_prefix}_state.pkl",
        resume=args.resume,
        seed=args.seed,
    )

    pt = ReplicaExchange.from_auto_config(**pt_kwargs)
    pt.run(n_cycles=args.n_cycles, equilibration_cycles=args.equilibration_cycles)


if __name__ == "__main__":
    main()
