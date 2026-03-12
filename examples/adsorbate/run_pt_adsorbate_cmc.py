from pathlib import Path
from types import SimpleNamespace

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


THIS_DIR = Path(__file__).resolve().parent
CONFIG = SimpleNamespace(
    snapshot=THIS_DIR.parent / "alloy" / "POSCAR.Ti2CO2",
    frame=0,
    repeat=(2, 2, 1),
    calculator="lj",
    model=None,
    device="cuda",
    lj_cutoff=6.0,
    adsorbate="OH",
    coverage=0.25,
    site_type="atop",
    surface_element="Ti",
    substrate_elements="Ti C",
    functional_elements=None,
    fix_below_z=None,
    move_mode="hybrid",
    site_hop_prob=0.35,
    reorientation_prob=0.35,
    rotation_max_angle_deg=25.0,
    displacement_sigma=0.6,
    max_displacement_trials=20,
    max_reorientation_trials=20,
    xy_tol=0.9,
    support_xy_tol=1.5,
    vertical_offset=1.8,
    relax=False,
    relax_steps=20,
    fmax=0.05,
    t_start=900.0,
    t_end=300.0,
    t_step=100.0,
    n_replicas=None,
    grid_space="temperature",
    n_cycles=20,
    equilibration_cycles=2,
    swap_interval=20,
    report_interval=5,
    sampling_interval=2,
    local_eq_fraction=0.0,
    checkpoint_interval=5,
    swap_stride=1,
    n_gpus=1,
    workers_per_gpu=1,
    execution_backend="multiprocessing",
    seed=81,
    output_prefix="adsorbate_pt",
    resume=False,
)


def main() -> None:
    atoms = load_snapshot(CONFIG.snapshot, CONFIG.frame)
    if tuple(CONFIG.repeat) != (1, 1, 1):
        atoms = atoms.repeat(tuple(CONFIG.repeat))
    maybe_fix_bottom(atoms, CONFIG.fix_below_z)

    substrate_elements = parse_symbols(CONFIG.substrate_elements)
    functional_elements = infer_functional_elements(
        atoms, substrate_elements, CONFIG.functional_elements
    )
    adsorbate_template, adsorbate_anchor_index = build_adsorbate_template(
        CONFIG.adsorbate
    )

    calculator = build_calculator(CONFIG)
    seed_builder = AdsorbateCMC.from_clean_surface(
        atoms=atoms,
        calculator=calculator,
        T=CONFIG.t_start,
        adsorbate=adsorbate_template,
        adsorbate_anchor_index=adsorbate_anchor_index,
        substrate_elements=substrate_elements,
        functional_elements=functional_elements,
        top_layer_element=CONFIG.surface_element,
        coverage=CONFIG.coverage,
        site_type=CONFIG.site_type,
        move_mode=CONFIG.move_mode,
        site_hop_prob=CONFIG.site_hop_prob,
        reorientation_prob=CONFIG.reorientation_prob,
        rotation_max_angle_deg=CONFIG.rotation_max_angle_deg,
        displacement_sigma=CONFIG.displacement_sigma,
        max_displacement_trials=CONFIG.max_displacement_trials,
        max_reorientation_trials=CONFIG.max_reorientation_trials,
        xy_tol=CONFIG.xy_tol,
        support_xy_tol=CONFIG.support_xy_tol,
        vertical_offset=CONFIG.vertical_offset,
        relax=CONFIG.relax,
        relax_steps=CONFIG.relax_steps,
        fmax=CONFIG.fmax,
        traj_file=f"{CONFIG.output_prefix}_seed.traj",
        thermo_file=f"{CONFIG.output_prefix}_seed.dat",
        checkpoint_file=f"{CONFIG.output_prefix}_seed.pkl",
        initial_traj_file=f"{CONFIG.output_prefix}_initial.traj",
        seed=CONFIG.seed,
    )

    calculator_class, calc_kwargs = get_calculator_class_and_kwargs(CONFIG)
    pt_kwargs = dict(
        atoms_template=seed_builder.atoms.copy(),
        T_start=CONFIG.t_start,
        T_end=CONFIG.t_end,
        T_step=CONFIG.t_step,
        n_replicas=CONFIG.n_replicas,
        grid_space=CONFIG.grid_space,
        calculator_class=calculator_class,
        calc_kwargs=calc_kwargs,
        mc_class=AdsorbateCMC,
        mc_kwargs={
            "adsorbate": adsorbate_template,
            "adsorbate_anchor_index": adsorbate_anchor_index,
            "substrate_elements": substrate_elements,
            "functional_elements": functional_elements,
            "top_layer_element": CONFIG.surface_element,
            "coverage": CONFIG.coverage,
            "site_type": CONFIG.site_type,
            "move_mode": CONFIG.move_mode,
            "site_hop_prob": CONFIG.site_hop_prob,
            "reorientation_prob": CONFIG.reorientation_prob,
            "rotation_max_angle_deg": CONFIG.rotation_max_angle_deg,
            "displacement_sigma": CONFIG.displacement_sigma,
            "max_displacement_trials": CONFIG.max_displacement_trials,
            "max_reorientation_trials": CONFIG.max_reorientation_trials,
            "xy_tol": CONFIG.xy_tol,
            "support_xy_tol": CONFIG.support_xy_tol,
            "vertical_offset": CONFIG.vertical_offset,
            "relax": CONFIG.relax,
            "relax_steps": CONFIG.relax_steps,
            "fmax": CONFIG.fmax,
            "checkpoint_interval": 0,
        },
        n_gpus=CONFIG.n_gpus,
        workers_per_gpu=CONFIG.workers_per_gpu,
        swap_interval=CONFIG.swap_interval,
        report_interval=CONFIG.report_interval,
        sampling_interval=CONFIG.sampling_interval,
        local_eq_fraction=CONFIG.local_eq_fraction,
        checkpoint_interval=CONFIG.checkpoint_interval,
        swap_stride=CONFIG.swap_stride,
        execution_backend=CONFIG.execution_backend,
        results_file=f"{CONFIG.output_prefix}_results.csv",
        stats_file=f"{CONFIG.output_prefix}_stats.csv",
        checkpoint_file=f"{CONFIG.output_prefix}_state.pkl",
        resume=CONFIG.resume,
        seed=CONFIG.seed,
    )

    pt = ReplicaExchange.from_auto_config(**pt_kwargs)
    pt.run(
        n_cycles=CONFIG.n_cycles,
        equilibration_cycles=CONFIG.equilibration_cycles,
    )


if __name__ == "__main__":
    main()
