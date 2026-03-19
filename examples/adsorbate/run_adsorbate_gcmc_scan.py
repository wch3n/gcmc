from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    # Used only when calculator='mace'. For parallel scans, list the actual devices you want to use.
    devices=("cuda:0",),
    lj_cutoff=6.0,
    adsorbate_element="H",
    temperature=300.0,
    mu_values=(-3.0, -2.0, -1.0, 0.0, 1.0),
    seeds=(81, 82),
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
    backend="multiprocessing",
    n_workers=1,
    ray_address=None,
    ray_log_to_driver=False,
    ray_num_cpus_per_task=1,
    ray_num_gpus_per_task=None,
    output_dir="adsorbate_gcmc_scan",
)


def _format_mu(mu: float) -> str:
    return f"{mu:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _run_one(task: dict) -> dict:
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimpleNamespace(**task["config"])
    cfg.device = task["device"]

    atoms = load_snapshot(Path(cfg.snapshot), int(cfg.frame))
    if tuple(cfg.repeat) != (1, 1, 1):
        atoms = atoms.repeat(tuple(cfg.repeat))
    maybe_fix_bottom(atoms, cfg.fix_below_z)
    calculator = build_calculator(cfg)

    substrate_elements = parse_symbols(cfg.substrate_elements)
    functional_elements = infer_functional_elements(
        atoms, substrate_elements, cfg.functional_elements
    )

    stem = task["stem"]
    sim = AdsorbateGCMC(
        atoms=atoms,
        calculator=calculator,
        mu=float(task["mu"]),
        T=cfg.temperature,
        nsteps=cfg.nsweeps,
        adsorbate_element=cfg.adsorbate_element,
        substrate_elements=substrate_elements,
        functional_elements=functional_elements,
        site_elements=parse_symbols(cfg.site_elements),
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
        traj_file=str(run_dir / f"{stem}.traj"),
        thermo_file=str(run_dir / f"{stem}.dat"),
        checkpoint_file=str(run_dir / f"{stem}.pkl"),
        checkpoint_interval=0,
        seed=int(task["seed"]),
    )

    stats = sim.run(
        nsweeps=cfg.nsweeps,
        traj_file=str(run_dir / f"{stem}.traj"),
        interval=cfg.interval,
        sample_interval=cfg.sample_interval,
        equilibration=cfg.equilibration,
    )
    return {
        "mu": float(task["mu"]),
        "seed": int(task["seed"]),
        "device": task["device"],
        "run_dir": str(run_dir),
        "traj_file": str(run_dir / f"{stem}.traj"),
        "thermo_file": str(run_dir / f"{stem}.dat"),
        "checkpoint_file": str(run_dir / f"{stem}.pkl"),
        **stats,
    }


def _ray_num_gpus_per_task() -> float:
    if CONFIG.ray_num_gpus_per_task is not None:
        return float(CONFIG.ray_num_gpus_per_task)
    return 1.0 if CONFIG.calculator == "mace" else 0.0


def _run_tasks_with_multiprocessing(tasks: list[dict]) -> list[dict]:
    results = []
    if CONFIG.n_workers == 1:
        for task in tasks:
            result = _run_one(task)
            results.append(result)
            print(
                f"done mu={result['mu']:+.3f} seed={result['seed']} "
                f"avgN={result['n_adsorbates_avg']:.3f} acc={result['acceptance']:.2f}%"
            )
        return results

    with ProcessPoolExecutor(max_workers=CONFIG.n_workers) as pool:
        future_map = {pool.submit(_run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(
                f"done mu={result['mu']:+.3f} seed={result['seed']} "
                f"avgN={result['n_adsorbates_avg']:.3f} acc={result['acceptance']:.2f}%"
            )
    return results


def _run_tasks_with_ray(tasks: list[dict]) -> list[dict]:
    import ray

    address = CONFIG.ray_address
    if address is None:
        address = os.getenv("RAY_ADDRESS")

    init_kwargs = {"log_to_driver": bool(CONFIG.ray_log_to_driver)}
    if address:
        init_kwargs["address"] = address

    ray.init(**init_kwargs)
    remote_run_one = ray.remote(
        num_cpus=float(CONFIG.ray_num_cpus_per_task),
        num_gpus=_ray_num_gpus_per_task(),
    )(_run_one)

    results = []
    pending = [remote_run_one.remote(task) for task in tasks]
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        result = ray.get(ready[0])
        results.append(result)
        print(
            f"done mu={result['mu']:+.3f} seed={result['seed']} "
            f"avgN={result['n_adsorbates_avg']:.3f} acc={result['acceptance']:.2f}%"
        )

    ray.shutdown()
    return results


def main() -> None:
    out_dir = (THIS_DIR / CONFIG.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = str(CONFIG.backend).lower()
    if backend not in {"multiprocessing", "ray"}:
        raise ValueError("CONFIG.backend must be 'multiprocessing' or 'ray'.")
    devices = tuple(CONFIG.devices) if getattr(CONFIG, "devices", None) else (None,)
    if backend == "multiprocessing" and CONFIG.n_workers < 1:
        raise ValueError("CONFIG.n_workers must be >= 1.")

    base_config = {
        key: value
        for key, value in CONFIG.__dict__.items()
        if key
        not in {
            "mu_values",
            "seeds",
            "devices",
            "n_workers",
            "backend",
            "ray_address",
            "ray_log_to_driver",
            "ray_num_cpus_per_task",
            "ray_num_gpus_per_task",
            "output_dir",
        }
    }

    tasks = []
    task_id = 0
    for mu in CONFIG.mu_values:
        mu_dir = out_dir / f"mu_{_format_mu(float(mu))}"
        for seed in CONFIG.seeds:
            device = (
                devices[task_id % len(devices)]
                if backend == "multiprocessing"
                else ("cuda" if _ray_num_gpus_per_task() > 0.0 else "cpu")
            )
            stem = f"mu_{_format_mu(float(mu))}_seed_{int(seed):03d}"
            tasks.append(
                {
                    "mu": float(mu),
                    "seed": int(seed),
                    "device": device,
                    "stem": stem,
                    "run_dir": str(mu_dir),
                    "config": base_config,
                }
            )
            task_id += 1

    print(f"Launching {len(tasks)} GCMC jobs at T={CONFIG.temperature:.1f} K")
    print(f"mu values: {tuple(float(mu) for mu in CONFIG.mu_values)}")
    print(f"seeds: {tuple(int(seed) for seed in CONFIG.seeds)}")
    print(f"backend: {backend}")
    if backend == "multiprocessing":
        print(f"workers: {CONFIG.n_workers}")
        print(f"devices: {devices}")
    else:
        print(f"ray address: {CONFIG.ray_address or os.getenv('RAY_ADDRESS', 'local')}")
        print(f"ray task cpus: {CONFIG.ray_num_cpus_per_task}")
        print(f"ray task gpus: {_ray_num_gpus_per_task()}")
    print(f"output dir: {out_dir}")

    if backend == "ray":
        results = _run_tasks_with_ray(tasks)
    else:
        results = _run_tasks_with_multiprocessing(tasks)

    results.sort(key=lambda row: (row["mu"], row["seed"]))
    summary_file = out_dir / "summary.csv"
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

    print(f"Wrote summary: {summary_file}")


if __name__ == "__main__":
    main()
