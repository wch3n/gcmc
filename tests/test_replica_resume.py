import os
import pickle
import signal
import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory

from gcmc.replica import ReplicaExchange
from gcmc.workflows import load_alloy_pt_config


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": [[0.0, 0.0, 0.0] for _ in atoms]}


class DummyMC:
    pass


class SignalAfterOneResultBackend:
    def __init__(self, pt):
        self.pt = pt
        self.started = False
        self.stopped = False
        self.submitted = 0

    def start(self):
        self.started = True

    def submit(self, replica_id, target_gpu, task_data):
        self.submitted += 1
        self.task_data = task_data

    def get_result(self):
        os.kill(os.getpid(), signal.SIGUSR1)
        state = self.pt.replica_states[0]
        return {
            "replica_id": 0,
            "positions": state["atoms"].get_positions(),
            "numbers": state["atoms"].get_atomic_numbers(),
            "tags": state["atoms"].get_tags(),
            "cell": state["atoms"].get_cell(),
            "pbc": state["atoms"].get_pbc(),
            "e_old": 0.0,
            "rng_state": state["rng_state"],
            "sweep": self.task_data["sweep"] + self.task_data["nsweeps"],
            "local_stats": {"energy": 0.0, "cv": 0.0, "acceptance": 100.0},
            "cycle_sum_E": 0.0,
            "cycle_sum_E_sq": 0.0,
            "cycle_n_samples": 1,
        }

    def stop(self):
        self.stopped = True


class ReplicaResumeTests(unittest.TestCase):
    def test_fresh_replica_states_defer_initial_energy_to_worker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pt = ReplicaExchange.from_auto_config(
                atoms_template=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
                T_start=300.0,
                T_end=400.0,
                T_step=100.0,
                calculator_class=ZeroCalculator,
                mc_class=DummyMC,
                n_gpus=1,
                workers_per_gpu=1,
                stats_file=str(root / "replica_stats.csv"),
                results_file=str(root / "results.csv"),
                checkpoint_file=str(root / "pt_state.pkl"),
            )

            self.assertEqual(
                [state["e_old"] for state in pt.replica_states], [None, None]
            )

    def test_resume_cleanup_truncates_outputs_to_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            traj_path = root / "replica_300K.traj"
            thermo_path = root / "replica_300K.dat"
            results_path = root / "results.csv"
            stats_path = root / "replica_stats.csv"

            atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            with Trajectory(str(traj_path), "w") as traj:
                for idx in range(5):
                    atoms.positions[0, 0] = float(idx)
                    traj.write(atoms)

            thermo_path.write_text(
                "5 -1.0\n10 -2.0\n15 -3.0\n20 -4.0\n25 -5.0\n"
            )
            results_path.write_text(
                "cycle,T_K,n_atoms,E_eV\n1,300,1,-1.0\n2,300,1,-2.0\n3,300,1,-3.0\n"
            )
            stats_path.write_text(
                "Cycle,T_i,T_j,E_i,E_j,Accepted\n0,300,400,-1,-2,True\n1,300,400,-2,-3,False\n2,300,400,-3,-4,True\n"
            )

            pt = ReplicaExchange(
                n_gpus=1,
                workers_per_gpu=1,
                replica_states=[
                    {
                        "id": 0,
                        "T": 300.0,
                        "atoms": atoms.copy(),
                        "e_old": -4.0,
                        "rng_state": None,
                        "traj_file": str(traj_path),
                        "thermo_file": str(thermo_path),
                        "checkpoint_file": str(root / "checkpoint_300K.pkl"),
                        "mc_kwargs": {},
                    }
                ],
                swap_interval=10,
                results_file=str(results_path),
                stats_file=str(stats_path),
                checkpoint_file=str(root / "pt_state.pkl"),
                resume=True,
            )
            pt.cycle_start = 2

            pt._prepare_resume_outputs()

            self.assertEqual(
                thermo_path.read_text(),
                "5 -1.0\n10 -2.0\n15 -3.0\n20 -4.0\n",
            )
            truncated_traj = Trajectory(str(traj_path))
            try:
                self.assertEqual(len(truncated_traj), 4)
            finally:
                truncated_traj.close()
            self.assertIn("2,300,1,-2.0", results_path.read_text())
            self.assertNotIn("3,300,1,-3.0", results_path.read_text())
            self.assertIn("1,300,400,-2,-3,False", stats_path.read_text())
            self.assertNotIn("2,300,400,-3,-4,True", stats_path.read_text())

    def test_sectioned_alloy_pt_checkpoint_intervals_are_not_collided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg_path = root / "config.yaml"
            cfg_path.write_text(
                """
system:
  snapshot: POSCAR
pt:
  checkpoint_interval: 1
mc:
  checkpoint_interval: 20
output:
  output_dir: out
""".strip()
            )

            cfg = load_alloy_pt_config(cfg_path)

        self.assertEqual(cfg.checkpoint_interval, 1)
        self.assertEqual(cfg.worker_checkpoint_interval, 20)

    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "SIGUSR1 is unavailable")
    def test_sigusr1_finishes_cycle_and_forces_master_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            pt = ReplicaExchange(
                n_gpus=1,
                workers_per_gpu=1,
                replica_states=[
                    {
                        "id": 0,
                        "T": 300.0,
                        "atoms": atoms,
                        "e_old": None,
                        "cum_sum_E": 0.0,
                        "cum_sum_E_sq": 0.0,
                        "cum_n_samples": 0,
                        "rng_state": None,
                        "traj_file": str(root / "replica_300K.traj"),
                        "thermo_file": str(root / "replica_300K.dat"),
                        "checkpoint_file": str(root / "checkpoint_300K.pkl"),
                        "mc_kwargs": {},
                    }
                ],
                swap_interval=1,
                checkpoint_interval=0,
                report_interval=0,
                results_file=str(root / "results.csv"),
                stats_file=str(root / "replica_stats.csv"),
                checkpoint_file=str(root / "pt_state.pkl"),
            )
            backend = SignalAfterOneResultBackend(pt)
            pt.backend = backend
            previous_handler = signal.getsignal(signal.SIGUSR1)

            pt.run(n_cycles=5)

            self.assertTrue(backend.started)
            self.assertTrue(backend.stopped)
            self.assertEqual(backend.submitted, 1)
            with (root / "pt_state.pkl").open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.assertEqual(checkpoint["cycle"], 1)
            self.assertEqual(signal.getsignal(signal.SIGUSR1), previous_handler)


if __name__ == "__main__":
    unittest.main()
