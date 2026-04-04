import unittest

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from gcmc.adsorbate_gcmc import AdsorbateGCMC


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


def _make_two_site_surface() -> Atoms:
    symbols = ["Ti", "Ti", "O", "O", "H", "H"]
    positions = [
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (0.0, 0.0, 1.2),
        (3.0, 0.0, 1.2),
        (0.0, 0.0, 2.2),
        (3.0, 0.0, 2.2),
    ]
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
        pbc=[True, True, False],
    )


class TestAdsorbateGCMCSiteAssignment(unittest.TestCase):
    def _make_sim(self) -> AdsorbateGCMC:
        return AdsorbateGCMC(
            atoms=_make_two_site_surface(),
            calculator=ZeroCalculator(),
            mu=-1.0,
            T=300.0,
            max_n_adsorbates=2,
            site_elements=("O",),
            substrate_elements=("Ti",),
            functional_elements=(),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.0,
            enable_hybrid_md=False,
            seed=7,
        )

    def test_site_assignment_valid_on_registry(self):
        sim = self._make_sim()
        self.assertTrue(sim._site_assignment_is_valid())

    def test_site_assignment_invalid_off_registry(self):
        sim = self._make_sim()
        atoms_trial = sim.atoms.copy()
        h_index = next(i for i, atom in enumerate(atoms_trial) if atom.symbol == "H")
        atoms_trial.positions[h_index, :2] = np.array([1.5, 0.0], dtype=float)
        self.assertFalse(sim._site_assignment_is_valid(atoms=atoms_trial))

    def test_deletion_proposal_rejected_from_invalid_state(self):
        sim = self._make_sim()
        h_index = next(i for i, atom in enumerate(sim.atoms) if atom.symbol == "H")
        sim.atoms.positions[h_index, :2] = np.array([1.5, 0.0], dtype=float)
        sim._refresh_cached_state()
        self.assertIsNone(sim._propose_deletion())


if __name__ == "__main__":
    unittest.main()
