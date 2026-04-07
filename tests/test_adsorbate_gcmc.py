import unittest

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator, all_changes

from gcmc.adsorbate_cmc import AdsorbateCMC
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


def _make_assignment_surface() -> Atoms:
    anchors = np.array(
        [
            [6.15385111, 3.83677554],
            [9.97209936, 9.80835339],
            [6.85541984, 6.50459276],
        ],
        dtype=float,
    )
    atoms = Atoms(
        symbols=["Ti", "Ti", "Ti"],
        positions=[
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (8.0, 0.0, 0.0),
        ],
        cell=[[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        pbc=[False, False, False],
    )
    for xy in anchors:
        atoms.append(Atom("H", (float(xy[0]), float(xy[1]), 2.0)))
    return atoms


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

    def test_assignment_uses_global_optimum(self):
        sim = AdsorbateGCMC(
            atoms=_make_assignment_surface(),
            calculator=ZeroCalculator(),
            mu=-1.0,
            T=300.0,
            max_n_adsorbates=3,
            site_elements=("Ti",),
            substrate_elements=("Ti",),
            functional_elements=(),
            site_type="atop",
            move_mode="site_hop",
            site_hop_prob=1.0,
            reorientation_prob=0.0,
            enable_hybrid_md=False,
            allow_ambiguous_empty_adsorbates=True,
            seed=11,
        )
        candidate_sites = [
            {"xy": np.array([6.88446731, 3.88921424], dtype=float)},
            {"xy": np.array([1.35096505, 7.21488340], dtype=float)},
            {"xy": np.array([5.25354322, 3.10241876], dtype=float)},
        ]
        mapping, _ = sim._assign_groups_to_sites_with_distances(
            candidate_sites=candidate_sites
        )
        self.assertEqual(mapping, {0: 2, 1: 1, 2: 0})


class TestAdsorbateCMCVerticalAdjustment(unittest.TestCase):
    def test_vertical_adjustment_recovers_valid_trial(self):
        slab = Atoms(
            symbols=["Ti", "O"],
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.2)],
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
            pbc=[False, False, False],
        )
        sim = AdsorbateCMC(
            atoms=slab,
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="H",
            adsorbate="H",
            substrate_elements=("Ti",),
            functional_elements=("O",),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="site_hop",
            site_hop_prob=1.0,
            reorientation_prob=0.0,
            min_clearance=0.8,
            termination_clearance=0.75,
            vertical_adjust_step=0.25,
            max_vertical_adjust=1.0,
            seed=5,
        )

        atoms_trial = slab.copy()
        atoms_trial.append(Atom("H", (0.0, 0.0, 1.35)))
        group = np.asarray([len(atoms_trial) - 1], dtype=int)
        adjusted = sim._adjust_trial_positions_vertically(
            group,
            atoms_trial.positions[group],
            atoms=atoms_trial,
        )

        self.assertIsNotNone(adjusted)
        self.assertGreater(float(adjusted[0, 2]), 1.35)
        self.assertTrue(
            sim._group_positions_are_valid(group, adjusted, atoms=atoms_trial)
        )
        self.assertTrue(
            sim._group_clears_terminations(group, adjusted, atoms=atoms_trial)
        )


if __name__ == "__main__":
    unittest.main()
