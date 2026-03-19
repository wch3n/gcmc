from pathlib import Path
from types import SimpleNamespace

from gcmc import AdsorbateGCMC
from common import (
    build_calculator,
    infer_functional_elements,
    load_snapshot,
    maybe_fix_bottom,
    parse_symbols,
)


THIS_DIR = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    snapshot=THIS_DIR.parent / "alloy" / "POSCAR.Ti2CO2",
    frame=0,
    repeat=(2, 2, 1),
    calculator="lj",
    model=None,
    device="cuda",
    lj_cutoff=6.0,
    adsorbate_element="H",
    temperature=300.0,
    chemical_potential=0.0,
    site_type=("atop", "bridge"),
    site_elements="Ti",
    substrate_elements="Ti C",
    functional_elements=None,
    fix_below_z=None,
    move_mode="hybrid",
    site_hop_prob=0.25,
    displacement_sigma=0.25,
    max_displacement_trials=20,
    min_clearance=0.9,
    site_match_tol=0.6,
    surface_layer_tol=0.5,
    termination_clearance=0.8,
    vertical_offset=1.8,
    w_insert=1.0,
    w_delete=1.0,
    w_canonical=1.0,
    max_n_adsorbates=None,
    relax=False,
    relax_steps=10,
    fmax=0.05,
    enable_hybrid_md=False,
    nsweeps=200,
    interval=10,
    sample_interval=2,
    equilibration=40,
    seed=81,
    output_prefix="adsorbate_gcmc",
)


def main() -> None:
    atoms = load_snapshot(CONFIG.snapshot, CONFIG.frame)
    if tuple(CONFIG.repeat) != (1, 1, 1):
        atoms = atoms.repeat(tuple(CONFIG.repeat))
    maybe_fix_bottom(atoms, CONFIG.fix_below_z)
    calculator = build_calculator(CONFIG)

    substrate_elements = parse_symbols(CONFIG.substrate_elements)
    functional_elements = infer_functional_elements(
        atoms, substrate_elements, CONFIG.functional_elements
    )

    sim = AdsorbateGCMC(
        atoms=atoms,
        calculator=calculator,
        mu=CONFIG.chemical_potential,
        T=CONFIG.temperature,
        nsteps=CONFIG.nsweeps,
        adsorbate_element=CONFIG.adsorbate_element,
        substrate_elements=substrate_elements,
        functional_elements=functional_elements,
        site_elements=parse_symbols(CONFIG.site_elements),
        site_type=CONFIG.site_type,
        move_mode=CONFIG.move_mode,
        site_hop_prob=CONFIG.site_hop_prob,
        displacement_sigma=CONFIG.displacement_sigma,
        max_displacement_trials=CONFIG.max_displacement_trials,
        min_clearance=CONFIG.min_clearance,
        site_match_tol=CONFIG.site_match_tol,
        surface_layer_tol=CONFIG.surface_layer_tol,
        termination_clearance=CONFIG.termination_clearance,
        vertical_offset=CONFIG.vertical_offset,
        w_insert=CONFIG.w_insert,
        w_delete=CONFIG.w_delete,
        w_canonical=CONFIG.w_canonical,
        max_n_adsorbates=CONFIG.max_n_adsorbates,
        relax=CONFIG.relax,
        relax_steps=CONFIG.relax_steps,
        fmax=CONFIG.fmax,
        enable_hybrid_md=CONFIG.enable_hybrid_md,
        traj_file=f"{CONFIG.output_prefix}.traj",
        thermo_file=f"{CONFIG.output_prefix}.dat",
        checkpoint_file=f"{CONFIG.output_prefix}.pkl",
        checkpoint_interval=0,
        seed=CONFIG.seed,
    )

    stats = sim.run(
        nsweeps=CONFIG.nsweeps,
        traj_file=f"{CONFIG.output_prefix}.traj",
        interval=CONFIG.interval,
        sample_interval=CONFIG.sample_interval,
        equilibration=CONFIG.equilibration,
    )

    print("Final AdsorbateGCMC stats:", stats)


if __name__ == "__main__":
    main()
