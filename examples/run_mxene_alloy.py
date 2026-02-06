#!/usr/bin/env python3

from ase.io import read
from ase.build import make_supercell

# from mace.calculators import MACECalculator
from symmetrix import Symmetrix
from gcmc.alloy_cmc import AlloyCMC
from gcmc.utils import initialize_alloy_sublattice, generate_temperature_grid
from gcmc.replica import ReplicaExchange
import numpy as np


def main(sc_matrix):

    # setup initial system
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

    mc_kwargs = {
        "swap_elements": ["Ti", "Mo"],
        "swap_mode": "hybrid",
        "relax": False,
        "fmax": 0.2,
        "relax_steps": 10,
        "checkpoint_interval": 1,
        "neighbor_cutoff": 5.0,
    }

    temps = generate_temperature_grid(
        T_start=2000,
        T_end=100,
        n_replicas=36,
        base="beta",
        windows=[(900, 500, 20), (500, 300, 10)],
    )

    """
    # initialize MC
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

    # Run cooling
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

    # replica exchange
    pt = ReplicaExchange.from_auto_config(
        atoms_template=atoms,
        temps=temps,
        swap_stride=1,
        calculator_class=Symmetrix,
        calc_kwargs=calc_kwargs,
        mc_class=AlloyCMC,
        mc_kwargs=mc_kwargs,
        swap_interval=20,
        report_interval=5,
        sampling_interval=5,
        checkpoint_interval=5,
        resume=True,
        n_gpus=4,
        workers_per_gpu=4,
    )

    pt.run(n_cycles=500, equilibration_cycles=20)


if __name__ == "__main__":
    sc_matrix = np.array([[8, 0, 0], [5, 10, 0], [0, 0, 1]])
    main(sc_matrix)
