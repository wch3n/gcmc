import unittest
import tempfile

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.process_replica import _restore_atoms_from_snapshot
from gcmc.replica import MuReplicaExchange


def _make_clean_slab() -> Atoms:
    atoms = Atoms(
        symbols=["Ti", "O"],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.8)],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=[False, False, False],
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms


def _with_n_adsorbates(base: Atoms, n_ads: int) -> Atoms:
    atoms = base.copy()
    tags = np.zeros(len(atoms), dtype=int)
    if len(atoms.get_tags()) == len(atoms):
        tags[:] = atoms.get_tags()
    for idx in range(n_ads):
        atoms.append("H")
        atoms.positions[-1] = (0.0, 0.0, 2.8 + idx)
        tags = np.append(tags, ADSORBATE_TAG_OFFSET + idx)
    atoms.set_tags(tags)
    return atoms


class StubRNG:
    def __init__(self, random_value: float):
        self.random_value = float(random_value)

    def integers(self, low, high=None):
        return 0

    def random(self):
        return self.random_value


class MuReplicaExchangeTests(unittest.TestCase):
    def test_restore_atoms_from_snapshot_preserves_constraints_and_tags(self):
        template = _make_clean_slab()
        trial = _with_n_adsorbates(template, 1)
        restored = _restore_atoms_from_snapshot(
            template,
            trial.get_positions(),
            trial.get_atomic_numbers(),
            trial.get_cell(),
            trial.get_pbc(),
            tags=trial.get_tags(),
        )

        self.assertEqual(len(restored), len(trial))
        self.assertTrue(np.array_equal(restored.get_tags(), trial.get_tags()))
        self.assertEqual(len(restored.constraints), 1)

    def test_mu_exchange_accepts_negative_delta_swap(self):
        base = _make_clean_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            states = [
                {
                    "id": 0,
                    "mu": -4.0,
                    "T": 300.0,
                    "atoms": _with_n_adsorbates(base, 0),
                    "e_old": -1.0,
                },
                {
                    "id": 1,
                    "mu": -5.0,
                    "T": 300.0,
                    "atoms": _with_n_adsorbates(base, 1),
                    "e_old": -1.2,
                },
            ]
            rex = MuReplicaExchange(
                n_gpus=1,
                workers_per_gpu=1,
                replica_states=states,
                worker_init_info={},
                stats_file=f"{tmpdir}/stats.csv",
                results_file=f"{tmpdir}/results.csv",
                checkpoint_file=f"{tmpdir}/state.pkl",
                seed=1,
            )
            rex.rng = StubRNG(random_value=1.0)
            before = [len(state["atoms"]) for state in rex.replica_states]
            rex._attempt_swaps(cycle=0)
            after = [len(state["atoms"]) for state in rex.replica_states]

            self.assertEqual(before, [2, 3])
            self.assertEqual(after, [3, 2])

    def test_mu_exchange_rejects_positive_delta_swap_when_random_is_one(self):
        base = _make_clean_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            states = [
                {
                    "id": 0,
                    "mu": -5.0,
                    "T": 300.0,
                    "atoms": _with_n_adsorbates(base, 0),
                    "e_old": -1.0,
                },
                {
                    "id": 1,
                    "mu": -4.0,
                    "T": 300.0,
                    "atoms": _with_n_adsorbates(base, 1),
                    "e_old": -1.2,
                },
            ]
            rex = MuReplicaExchange(
                n_gpus=1,
                workers_per_gpu=1,
                replica_states=states,
                worker_init_info={},
                stats_file=f"{tmpdir}/stats.csv",
                results_file=f"{tmpdir}/results.csv",
                checkpoint_file=f"{tmpdir}/state.pkl",
                seed=1,
            )
            rex.rng = StubRNG(random_value=1.0)
            before = [len(state["atoms"]) for state in rex.replica_states]
            rex._attempt_swaps(cycle=0)
            after = [len(state["atoms"]) for state in rex.replica_states]

            self.assertEqual(before, [2, 3])
            self.assertEqual(after, [2, 3])


if __name__ == "__main__":
    unittest.main()
