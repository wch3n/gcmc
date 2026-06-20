import multiprocessing
import os
import logging
import traceback
import importlib
from copy import deepcopy
import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

ctx = multiprocessing.get_context("spawn")


def _restore_atoms_from_snapshot(
    atoms_template: Atoms,
    positions,
    numbers,
    cell,
    pbc,
    tags=None,
) -> Atoms:
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    if tags is not None:
        atoms.set_tags(np.asarray(tags, dtype=int))
    template_constraints = getattr(atoms_template, "constraints", None)
    if template_constraints:
        atoms.set_constraint(deepcopy(template_constraints))
    return atoms


def _snapshot_matches_atoms(atoms: Atoms, data) -> bool:
    """Return True when an incoming snapshot is identical to the current atoms state."""
    if len(atoms) != len(data["positions"]):
        return False
    if not np.array_equal(atoms.get_positions(), np.asarray(data["positions"], dtype=float)):
        return False
    if not np.array_equal(
        atoms.get_atomic_numbers(), np.asarray(data["numbers"], dtype=int)
    ):
        return False
    incoming_tags = data.get("tags")
    if incoming_tags is not None and not np.array_equal(
        atoms.get_tags(), np.asarray(incoming_tags, dtype=int)
    ):
        return False
    if not np.array_equal(atoms.get_cell().array, np.asarray(data["cell"], dtype=float)):
        return False
    if not np.array_equal(atoms.get_pbc(), np.asarray(data["pbc"], dtype=bool)):
        return False
    return True


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
            # Dynamic import.
            mc_module_name = self.init_kwargs.get("mc_module", "gcmc.alloy_cmc")
            mc_class_name = self.init_kwargs.get("mc_class", "AlloyCMC")

            mc_module = importlib.import_module(mc_module_name)
            MCClass = getattr(mc_module, mc_class_name)

            # Import calculator.
            calc_module_name = self.init_kwargs["calculator_module"]
            calc_class_name = self.init_kwargs["calculator_class_name"]
            calc_module = importlib.import_module(calc_module_name)
            CalcClass = getattr(calc_module, calc_class_name)

            calc_args = self.init_kwargs.get("calc_kwargs", {}).copy()
            if "device" in calc_args and "cuda" in str(calc_args["device"]):
                calc_args["device"] = "cuda:0"

            calc = CalcClass(**calc_args)
            atoms_template = self.init_kwargs["atoms_template"].copy()

            # Initialize engine with placeholder file paths.
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

                # Update state.
                sim.T = data["T"]
                if "mu" in data:
                    sim.mu = data["mu"]

                snapshot_changed = not _snapshot_matches_atoms(sim.atoms, data)

                # Safely refresh atom container when atom count changes.
                if len(sim.atoms) != len(data["positions"]):
                    sim.atoms = _restore_atoms_from_snapshot(
                        atoms_template,
                        data["positions"],
                        data["numbers"],
                        data["cell"],
                        data["pbc"],
                        tags=data.get("tags"),
                    )
                else:
                    sim.atoms.set_positions(data["positions"])
                    sim.atoms.set_atomic_numbers(data["numbers"])
                    if "tags" in data:
                        sim.atoms.set_tags(data["tags"])
                    sim.atoms.set_cell(data["cell"])
                    sim.atoms.pbc = data["pbc"]

                if snapshot_changed:
                    sim._refresh_cached_state()

                if data.get("e_old") is not None:
                    sim.e_old = data["e_old"]
                else:
                    sim.e_old = sim.get_potential_energy()
                if "rng_state" in data and data["rng_state"] is not None:
                    sim.rng.bit_generator.state = data["rng_state"]
                sim.sweep = data["sweep"]

                # Update file paths for this task chunk.
                sim.traj_file = data["traj_file"]
                sim.thermo_file = data["thermo_file"]
                sim.checkpoint_file = data["checkpoint_file"]
                if hasattr(sim, "attempted_traj_file"):
                    sim.attempted_traj_file = data.get("attempted_traj_file")
                if hasattr(sim, "accepted_traj_file"):
                    sim.accepted_traj_file = data.get("accepted_traj_file")
                if hasattr(sim, "rejected_traj_file"):
                    sim.rejected_traj_file = data.get("rejected_traj_file")

                if hasattr(sim, "tree"):
                    sim.tree = cKDTree(data["positions"])

                # Run simulation chunk.
                stats = sim.run(
                    nsweeps=data["nsweeps"],
                    traj_file=data["traj_file"],
                    interval=data["report_interval"],
                    sample_interval=data["sample_interval"],
                    equilibration=data["eq_steps"],
                    sweeps_are_total=False,
                )

                # Return results.
                result_package = {
                    "replica_id": replica_id,
                    "positions": sim.atoms.get_positions(),
                    "numbers": sim.atoms.get_atomic_numbers(),
                    "tags": sim.atoms.get_tags(),
                    "cell": sim.atoms.get_cell(),
                    "pbc": sim.atoms.get_pbc(),
                    "e_old": sim.e_old,
                    "rng_state": sim.rng.bit_generator.state,
                    "sweep": sim.sweep,
                    "cycle_sum_E": sim.sum_E,
                    "cycle_sum_E_sq": sim.sum_E_sq,
                    "cycle_sum_N": getattr(sim, "sum_N", 0.0),
                    "cycle_sum_N_sq": getattr(sim, "sum_N_sq", 0.0),
                    "cycle_n_samples": sim.n_samples,
                    "local_stats": stats,
                }
                self.result_queue.put(result_package)

        except Exception as e:
            traceback.print_exc()
            self.result_queue.put(("ERROR", str(e)))
