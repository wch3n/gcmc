# gcmc/process_replica.py

import multiprocessing
import os
import logging
import traceback
import importlib
import numpy as np
from scipy.spatial import cKDTree

ctx = multiprocessing.get_context("spawn")


class ReplicaWorker(ctx.Process):
    def __init__(self, rank, device_id, task_queue, result_queue, init_kwargs):
        super().__init__()
        self.rank = rank
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.init_kwargs = init_kwargs
        self.daemon = True

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        logging.basicConfig(
            format=f"[GPU {self.device_id}|W {self.rank}] %(message)s",
            level=logging.INFO,
        )

        try:
            # Dynamic Import
            mc_module_name = self.init_kwargs.get("mc_module", "gcmc.alloy_cmc")
            mc_class_name = self.init_kwargs.get("mc_class", "AlloyCMC")

            mc_module = importlib.import_module(mc_module_name)
            MCClass = getattr(mc_module, mc_class_name)

            # Import Calculator
            calc_module_name = self.init_kwargs["calculator_module"]
            calc_class_name = self.init_kwargs["calculator_class_name"]
            calc_module = importlib.import_module(calc_module_name)
            CalcClass = getattr(calc_module, calc_class_name)

            calc_args = self.init_kwargs.get("calc_kwargs", {}).copy()
            if "device" in calc_args and "cuda" in str(calc_args["device"]):
                calc_args["device"] = "cuda:0"

            calc = CalcClass(**calc_args)
            atoms_template = self.init_kwargs["atoms_template"].copy()

            # Initialize Engine with PLACEHOLDERS
            sim = MCClass(
                atoms=atoms_template,
                calculator=calc,
                T=300,
                traj_file="placeholder.traj",
                thermo_file="placeholder.dat",
                checkpoint_file="placeholder.pkl",
                **self.init_kwargs["mc_kwargs"],
            )

            while True:
                task = self.task_queue.get()
                if task == "STOP":
                    break

                replica_id, data = task

                # --- UPDATE STATE ---
                sim.T = data["T"]

                # Safe atoms update
                if len(sim.atoms) != len(data["positions"]):
                    sim.atoms = atoms_template.copy()

                sim.atoms.set_positions(data["positions"])
                sim.atoms.set_atomic_numbers(data["numbers"])
                sim.atoms.set_cell(data["cell"])
                sim.atoms.pbc = data["pbc"]

                sim.e_old = data["e_old"]
                if "rng_state" in data and data["rng_state"] is not None:
                    sim.rng.bit_generator.state = data["rng_state"]
                sim.sweep = data["sweep"]

                # --- FIX: UPDATE FILE PATHS ---
                # This was missing! The worker must update these for every chunk.
                sim.traj_file = data["traj_file"]
                sim.thermo_file = data["thermo_file"]
                sim.checkpoint_file = data["checkpoint_file"]

                if hasattr(sim, "tree"):
                    sim.tree = cKDTree(data["positions"])

                # Run
                stats = sim.run(
                    nsweeps=data["nsweeps"],
                    traj_file=data["traj_file"],
                    interval=data["report_interval"],
                    sample_interval=data["sample_interval"],
                    equilibration=data["eq_steps"],
                )

                # Return Results
                result_package = {
                    "replica_id": replica_id,
                    "positions": sim.atoms.get_positions(),
                    "numbers": sim.atoms.get_atomic_numbers(),
                    "cell": sim.atoms.get_cell(),
                    "pbc": sim.atoms.get_pbc(),
                    "e_old": sim.e_old,
                    "rng_state": sim.rng.bit_generator.state,
                    "sweep": sim.sweep,
                    "cycle_sum_E": sim.sum_E,
                    "cycle_sum_E_sq": sim.sum_E_sq,
                    "cycle_n_samples": sim.n_samples,
                    "local_stats": stats,
                }
                self.result_queue.put(result_package)

        except Exception as e:
            traceback.print_exc()
            self.result_queue.put(("ERROR", str(e)))
