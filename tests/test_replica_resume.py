import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import Trajectory

from gcmc.replica import ReplicaExchange
from gcmc.workflows import load_alloy_pt_config


class ReplicaResumeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
