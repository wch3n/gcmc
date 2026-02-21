#!/usr/bin/env python3

import os
import numpy as np
from ase.build import make_supercell
from ase.io import read

from symmetrix import Symmetrix
from gcmc.alloy_cmc import AlloyCMC
from gcmc.replica import ReplicaExchange
from gcmc.utils import (
    generate_nonuniform_temperature_grid,
    initialize_alloy_sublattice,
)


def main(sc_matrix):
    # Set up initial system.
    pristine = read("POSCAR.Ti2CO2")
    sc = make_supercell(pristine, sc_matrix)
    atoms = initialize_alloy_sublattice(
        atoms=sc,
        site_element="Ti",
        composition={"Ti": 0.5, "Mo": 0.5},
        seed=67,
    )

    calc_kwargs = {
        "model_file": "/gpfs/scratch/acad/htbase/wchen/mxene/mlip/pbe0-rvv10_8/Ti-Zr-Mo/models/ft-omat_0_sm_multi-00_stagetwo-1-22-40-42-6-8.json",
        "use_kokkos": True,
    }

    mc_kwargs = {
        "swap_elements": ["Ti", "Mo"],
        "swap_mode": "hybrid",
        "hybrid_neighbor_prob": 0.8,
        "neighbor_backend": "auto",
        "neighbor_cache": True,
        "relax": False,
        "fmax": 0.2,
        "relax_steps": 10,
        "checkpoint_interval": 1,
        "neighbor_cutoff": 5.0,
    }

    # Cluster topology.
    # Default matches the provided SLURM script (2 nodes x 4 GPUs/node).
    n_nodes = int(os.getenv("PT_NODES", "2"))
    gpus_per_node = int(os.getenv("PT_GPUS_PER_NODE", "4"))
    workers_per_gpu = int(os.getenv("PT_WORKERS_PER_GPU", "2"))
    total_gpu_slots = n_nodes * gpus_per_node

    # PT temperature-grid setup.
    use_nonuniform_grid = True

    pt_kwargs = dict(
        atoms_template=atoms,
        T_start=800,
        T_end=50,
        T_step=50,  # Used only when n_replicas is not set.
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
        n_gpus=total_gpu_slots,
        workers_per_gpu=workers_per_gpu,
        execution_backend="ray",
        backend_kwargs={
            "init_kwargs": {
                "address": "auto",
                "log_to_driver": False,
            },
            # Keep placement explicit to spread actor bundles across nodes.
            "use_placement_group": True,
            "placement_group_strategy": "SPREAD",
        },
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

    print(
        f"Ray PT config: n_gpus={total_gpu_slots}, workers_per_gpu={workers_per_gpu}, "
        f"total_workers={total_gpu_slots * workers_per_gpu}"
    )

    pt = ReplicaExchange.from_auto_config(**pt_kwargs)
    pt.run(n_cycles=2, equilibration_cycles=0)


if __name__ == "__main__":
    sc_matrix = np.array([[8, 0, 0], [5, 10, 0], [0, 0, 1]])
    main(sc_matrix)
