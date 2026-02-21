#!/usr/bin/env python3

from ase.io import read
from ase.build import make_supercell

# Optional alternative: from mace.calculators import MACECalculator.
from symmetrix import Symmetrix
from gcmc.alloy_cmc import AlloyCMC
from gcmc.utils import (
    initialize_alloy_sublattice,
    generate_nonuniform_temperature_grid,
)
from gcmc.replica import ReplicaExchange
import numpy as np


def main(sc_matrix):

    # Set up initial system.
    pristine = read("POSCAR.Ti2CO2")
    sc = make_supercell(pristine, sc_matrix)
    atoms = initialize_alloy_sublattice(
        atoms=sc,
        site_element="Ti",
        composition={"Ti": 1.0 / 2, "Mo": 1.0 / 2},
        seed=67,
    )

    calc_kwargs = {
        "model_file": "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti-Zr-Mo/models/ft-omat_0_sm_multi-00_stagetwo-1-22-40-42-6-8.json",
        "use_kokkos": True,
    }

    calc_kwargs_mace = {
        "model_paths": [
            "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti-Zr-Mo/models/ft-omat_0_sm_multi-00_stagetwo.model"
        ],
        "device": "cuda",
        "default_dtype": "float64",
    }

    neighbor_backend = "auto"  # Use matscipy if installed, otherwise ASE.
    neighbor_cache = True  # Set False to force neighbor rebuild every proposal.

    mc_kwargs = {
        "swap_elements": ["Ti", "Mo"],
        "swap_mode": "hybrid",
        "hybrid_neighbor_prob": 0.8,
        "neighbor_backend": neighbor_backend,
        "neighbor_cache": neighbor_cache,
        "relax": False,
        "fmax": 0.2,
        "relax_steps": 10,
        "checkpoint_interval": 1,
        "neighbor_cutoff": 5.0,
    }

    # Temperature-grid setup for replica exchange.
    # Switch to False to use a uniform grid controlled by T_step.
    use_nonuniform_grid = True

    """
    # Initialize MC.
    mc = AlloyCMC(
        atoms=atoms.repeat([3,3,1]),
        calculator=calc,
        swap_elements=['Ti', 'Mo', 'Zr'],
        swap_mode='hybrid', 
        relax=False,
        relax_steps=10,
        resume=True,
        checkpoint_file="restart.pkl"
    )

    # Run cooling.
    mc.run_temperature_scan(
        T_start=2000,
        T_end=100,
        T_step=10,
        sweeps_per_temp=1000, 
        equilibration=100,       
        interval=2,
        scan_file='results.csv',
        traj_prefix='run'
    )"""

    # Execution backend for replica workers: "multiprocessing" (default) or "ray".
    execution_backend = "multiprocessing"

    # Run replica exchange.
    pt_kwargs = dict(
        atoms_template=atoms,
        T_start=800,
        T_end=50,
        T_step=50,  # Used for uniform grid and ignored when n_replicas is set.
        swap_stride=1,
        calculator_class=Symmetrix,
        calc_kwargs=calc_kwargs,
        mc_class=AlloyCMC,
        mc_kwargs=mc_kwargs,
        swap_interval=20,
        report_interval=5,
        sampling_interval=5,
        local_eq_fraction=0.0,
        checkpoint_interval=5,
        resume=True,
        n_gpus=4,
        workers_per_gpu=4,
        execution_backend=execution_backend,
    )

    if use_nonuniform_grid:
        pt_kwargs.update(
            {
                "n_replicas": 16,
                "fine_grid_temps": [650.0, 300.0],
                "fine_grid_weights": [1.0, 1.5],
                "fine_grid_strength": 5.0,
                "fine_grid_width": 80.0,
                "grid_space": "beta",
            }
        )
        preview_temps = generate_nonuniform_temperature_grid(
            T_start=pt_kwargs["T_start"],
            T_end=pt_kwargs["T_end"],
            n_replicas=pt_kwargs["n_replicas"],
            focus_temps=pt_kwargs["fine_grid_temps"],
            focus_weights=pt_kwargs["fine_grid_weights"],
            focus_strength=pt_kwargs["fine_grid_strength"],
            focus_width=pt_kwargs["fine_grid_width"],
            grid_space=pt_kwargs["grid_space"],
        )
        print(
            "Nonuniform PT temperatures [K]:",
            np.array2string(np.array(preview_temps), precision=1),
        )

    pt = ReplicaExchange.from_auto_config(**pt_kwargs)

    pt.run(n_cycles=2, equilibration_cycles=0)


if __name__ == "__main__":
    sc_matrix = np.array([[8, 0, 0], [5, 10, 0], [0, 0, 1]])
    main(sc_matrix)
