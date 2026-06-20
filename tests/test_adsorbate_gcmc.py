import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atom, Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.io.trajectory import Trajectory

from gcmc.adsorbate_cmc import AdsorbateCMC, _place_adsorbate_template
from gcmc.adsorbate_move_proposals import MoveProposal
from gcmc.adsorbate_gcmc import AdsorbateGCMC
from gcmc.alloy_cmc import AlloyCMC
from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.utils import place_adsorbate_on_site


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


def _make_oh_surface() -> Atoms:
    atoms = Atoms(
        symbols=["Ti", "O", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.8),
            (0.0, 0.0, 2.78),
        ],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=[False, False, False],
    )
    atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
    return atoms


def _make_wrapped_oh_surface() -> Atoms:
    atoms = Atoms(
        symbols=["Ti", "O", "H"],
        positions=[
            (5.0, 5.0, 0.0),
            (0.1, 5.0, 1.8),
            (9.9, 5.0, 1.8),
        ],
        cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
        pbc=[True, True, False],
    )
    atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
    return atoms


def _make_ti_zr_alloy() -> Atoms:
    return Atoms(
        symbols=["Ti", "Zr"],
        positions=[(0.0, 0.0, 0.0), (2.5, 0.0, 0.0)],
        cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        pbc=[False, False, False],
    )


class StubRNG:
    def __init__(self, axis, angle):
        self.axis = np.asarray(axis, dtype=float)
        self.angle = float(angle)

    def choice(self, values):
        return values[0]

    def normal(self, size=None):
        if size is None:
            raise ValueError("StubRNG.normal requires size for this test.")
        return self.axis.copy()

    def uniform(self, low, high):
        return self.angle

    def permutation(self, value):
        return np.arange(value)


class FixedDisplacementRNG:
    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)

    def choice(self, values):
        return values[0]

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is not None:
            return np.resize(self.delta, size)
        return self.delta.copy()

    def uniform(self, low, high):
        return float(low)

    def permutation(self, value):
        return np.arange(value)


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


class TestAdsorbateCMCGeometry(unittest.TestCase):
    def _make_sim(self, atoms: Atoms, **overrides) -> AdsorbateCMC:
        data = dict(
            atoms=atoms,
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="displacement",
            min_clearance=0.1,
            termination_clearance=0.0,
            seed=13,
        )
        data.update(overrides)
        return AdsorbateCMC(**data)

    def test_tilted_cell_support_dz_uses_mic_displacement(self):
        atoms = Atoms(
            "TiOH",
            positions=[
                (9.9, 0.0, 0.0),
                (0.1, 0.0, 1.8),
                (0.1, 0.0, 2.78),
            ],
            cell=[[10.0, 0.0, 3.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(atoms, support_xy_tol=0.5, z_max_support=2.5)
        group = np.asarray(sim.ads_groups[0], dtype=int)

        self.assertIsNone(sim._nearest_support_atom_for_anchor(group))
        self.assertTrue(sim.has_afloat_adsorbates())

    def test_tilted_cell_same_site_uses_projected_xy_mic(self):
        atoms = Atoms(
            "TiOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 1.8),
                (0.1, 0.0, 2.78),
            ],
            cell=[[10.0, 0.0, 3.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(atoms, same_site_tol=0.5)

        self.assertTrue(
            sim._same_site_as_current(
                np.array([0.1, 0.0, 1.8], dtype=float),
                np.array([9.9, 0.0], dtype=float),
            )
        )

    def test_displacement_support_height_excludes_other_adsorbates(self):
        atoms = Atoms(
            "Ti2OHOH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
                (3.4, 0.0, 2.5),
                (3.4, 0.0, 3.48),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 15.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags(
            [
                0,
                0,
                ADSORBATE_TAG_OFFSET,
                ADSORBATE_TAG_OFFSET,
                ADSORBATE_TAG_OFFSET + 1,
                ADSORBATE_TAG_OFFSET + 1,
            ]
        )
        sim = self._make_sim(
            atoms,
            support_xy_tol=0.75,
            displacement_sigma=1.0,
            max_displacement_trials=1,
        )
        sim.rng = FixedDisplacementRNG([3.0, 0.0])

        trial = sim._propose_displacement()

        self.assertIsNotNone(trial)
        moved_group = np.asarray(sim.ads_groups[0], dtype=int)
        anchor_idx = int(moved_group[0])
        self.assertAlmostEqual(trial.positions[anchor_idx, 2], sim.vertical_offset)

    def test_initial_placement_uses_site_support_height_before_terminations(self):
        slab = Atoms(
            "MoO",
            positions=[
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 2.0),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        site = {
            "xy": np.array([0.0, 0.0], dtype=float),
            "surface_side": "top",
            "support_indices": np.array([0], dtype=int),
            "anchor_z_A": 2.0,
            "suggested_z_A": 3.5,
            "blocked_by_termination": False,
        }
        placed = _place_adsorbate_template(
            slab,
            Atoms("O", positions=[(0.0, 0.0, 0.0)]),
            anchor_index=0,
            site_registry=[site],
            coverage=1.0,
            seed=1,
        )

        self.assertAlmostEqual(placed.positions[-1, 2], 1.5, places=10)

    def test_place_adsorbate_on_site_uses_site_support_height(self):
        slab = Atoms(
            "MoO",
            positions=[
                (0.0, 0.0, 0.0),
                (0.1, 0.0, 2.0),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        site = {
            "xy": np.array([0.0, 0.0], dtype=float),
            "surface_side": "top",
            "support_indices": np.array([0], dtype=int),
            "anchor_z_A": 2.0,
            "suggested_z_A": 3.5,
        }

        placed, support_indices = place_adsorbate_on_site(
            slab,
            Atoms("O", positions=[(0.0, 0.0, 0.0)]),
            site,
            anchor_index=0,
        )

        self.assertEqual(support_indices.tolist(), [0])
        self.assertAlmostEqual(placed.positions[-1, 2], 1.5, places=10)

    def test_place_adsorbate_on_site_can_use_oo_center_anchor(self):
        slab = Atoms(
            "Mo",
            positions=[(0.0, 0.0, 0.0)],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        site = {
            "xy": np.array([0.0, 0.0], dtype=float),
            "surface_side": "top",
            "support_indices": np.array([0], dtype=int),
            "anchor_z_A": 0.0,
            "suggested_z_A": 1.5,
        }
        template = Atoms(
            "OOH",
            positions=[
                (-0.5, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (0.7, 0.8, 0.2),
            ],
        )

        placed, _ = place_adsorbate_on_site(
            slab,
            template,
            site,
            anchor_index=0,
            anchor_mode="center_of_mass",
            anchor_atom_indices=[0, 1],
        )

        oo_center = np.mean(placed.positions[-3:-1], axis=0)
        np.testing.assert_allclose(oo_center, np.array([0.0, 0.0, 1.5]), atol=1e-12)

    def test_template_library_samples_flat_and_upright_relative_positions(self):
        atoms = Atoms(
            "MoOOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.5),
                (0.0, 0.0, 2.96),
                (0.79, 0.0, 3.51),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 14.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        upright = Atoms(
            "OOH",
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.46), (0.79, 0.0, 2.01)],
        )
        flat = Atoms(
            "OOH",
            positions=[(-0.65, 0.0, 0.0), (0.65, 0.0, 0.0), (0.9, 0.8, 0.2)],
        )
        sim = self._make_sim(
            atoms,
            adsorbate=upright,
            adsorbate_template_library=[
                {
                    "adsorbate": upright,
                    "weight": 1.0,
                    "anchor_index": 0,
                },
                {
                    "adsorbate": flat,
                    "weight": 1.0,
                    "anchor_index": 0,
                    "anchor": {
                        "mode": "center_of_mass",
                        "atom_indices": [0, 1],
                    },
                },
            ],
        )

        samples = [sim._sample_template_group_relative_positions() for _ in range(80)]
        oo_gaps = sorted({round(abs(float(sample[1, 2] - sample[0, 2])), 2) for sample in samples})
        self.assertIn(0.0, oo_gaps)
        self.assertIn(1.46, oo_gaps)

    def test_template_library_accepts_builtin_ooh_string(self):
        atoms = Atoms(
            "MoOOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.5),
                (0.0, 0.0, 2.96),
                (0.79, 0.0, 3.51),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 14.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])

        sim = self._make_sim(
            atoms,
            adsorbate=Atoms(
                "OOH",
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.46), (0.79, 0.0, 2.01)],
            ),
            adsorbate_template_library=[
                {
                    "adsorbate": "OOH",
                    "weight": 1.0,
                    "anchor": {"mode": "atom", "atom_index": 0},
                }
            ],
        )

        self.assertEqual(len(sim.adsorbate_template_library), 1)
        self.assertEqual(sim.adsorbate_template_library[0]["template"].get_chemical_formula(), "HO2")

    def test_site_hop_uses_registry_suggested_height(self):
        atoms = Atoms(
            "Ti2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 15.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms,
            move_mode="site_hop",
            support_xy_tol=1.2,
            z_max_support=6.0,
        )
        sim._site_registry = [
            {
                "xy": np.array([0.0, 0.0], dtype=float),
                "site_type": "atop",
                "support_indices": np.array([0], dtype=int),
                "suggested_z_A": 1.8,
                "blocked_by_termination": False,
            },
            {
                "xy": np.array([3.0, 0.0], dtype=float),
                "site_type": "atop",
                "support_indices": np.array([1], dtype=int),
                "suggested_z_A": 5.0,
                "blocked_by_termination": False,
            },
        ]

        trial = sim._propose_site_hop()

        self.assertIsNotNone(trial)
        group = np.asarray(sim.ads_groups[0], dtype=int)
        anchor_idx = int(group[0])
        self.assertAlmostEqual(trial.positions[anchor_idx, 0], 3.0, places=10)
        self.assertAlmostEqual(trial.positions[anchor_idx, 2], 5.0, places=10)
        self.assertFalse(sim.has_afloat_adsorbates(trial))


class _FakeTrajectory:
    created = []

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.closed = False
        self.records = []
        _FakeTrajectory.created.append(self)

    def write(self, atoms):
        self.records.append(len(atoms))

    def close(self):
        self.closed = True


class TestAdsorbateCMCPersistentIO(unittest.TestCase):
    def test_metropolis_rejected_debug_frame_records_energy_metadata(self):
        sim = AdsorbateCMC(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate=Atoms(
                symbols=["O", "H"],
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)],
            ),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="hybrid",
            seed=19,
        )
        sim.e_old = -1.0
        sim._metropolis_accept = lambda *args, **kwargs: False
        proposal = MoveProposal(atoms=sim.atoms.copy(), move_name="site_hop")

        with tempfile.TemporaryDirectory() as tmpdir:
            rejected_path = Path(tmpdir) / "rejected.traj"
            writer = Trajectory(str(rejected_path), "w")
            sim._process_move_proposal(
                proposal,
                beta=1.0,
                move_ind=(0, 0),
                write_debug_frame=True,
                attempted_writer=None,
                accepted_writer=None,
                rejected_writer=writer,
            )
            writer.close()

            frame = read(str(rejected_path))

        self.assertEqual(frame.info["mc_event"], "rejected")
        self.assertEqual(frame.info["mc_reject_reason"], "metropolis")
        self.assertAlmostEqual(frame.info["mc_energy_eV"], 0.0)
        self.assertAlmostEqual(frame.info["mc_current_energy_eV"], -1.0)
        self.assertAlmostEqual(frame.info["mc_delta_e_eV"], 1.0)
        self.assertAlmostEqual(frame.info["mc_acceptance_delta_eV"], 1.0)
        self.assertAlmostEqual(frame.info["mc_accept_prob"], np.exp(-1.0))
        self.assertFalse(frame.info["mc_accepted"])
        self.assertEqual(frame.info["mc_acceptance_mode"], "potential")

    def test_nonconverged_relaxation_does_not_block_acceptance(self):
        sim = AdsorbateCMC(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate=Atoms(
                symbols=["O", "H"],
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)],
            ),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="hybrid",
            relax=True,
            diagnostics_enabled=True,
            seed=17,
        )
        sim.relax_structure = lambda atoms, move_ind=None: (atoms.copy(), False)
        proposal = MoveProposal(
            atoms=sim.atoms.copy(),
            move_name="site_hop",
            metadata={"test": "nonconverged"},
        )

        sim._process_move_proposal(
            proposal,
            beta=1.0,
            move_ind=(0, 0),
            write_debug_frame=False,
            attempted_writer=None,
            accepted_writer=None,
            rejected_writer=None,
        )

        self.assertEqual(sim.accepted_moves, 1)
        self.assertEqual(sim.move_diagnostics["accepted_by_move"]["site_hop"], 1)
        self.assertEqual(sim.move_diagnostics["rejected_by_reason"], {})

    def test_reuse_io_keeps_traj_writer_open_across_chunks(self):
        sim = AdsorbateCMC(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate=Atoms(
                symbols=["O", "H"],
                positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)],
            ),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.5,
            enable_hybrid_md=False,
            seed=9,
        )
        sim._reuse_io = True
        sim._propose_move = lambda: None
        sim._save_checkpoint = lambda: None

        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = str(Path(tmpdir) / "chunked.traj")
            thermo_file = str(Path(tmpdir) / "chunked.dat")
            sim.thermo_file = thermo_file

            _FakeTrajectory.created = []
            with mock.patch("gcmc.adsorbate_cmc.Trajectory", _FakeTrajectory):
                sim.run(nsweeps=1, traj_file=traj_file, interval=1, sample_interval=1)
                sim.run(nsweeps=2, traj_file=traj_file, interval=1, sample_interval=1)

            self.assertEqual(len(_FakeTrajectory.created), 1)
            self.assertEqual(_FakeTrajectory.created[0].filename, traj_file)
            self.assertEqual(len(_FakeTrajectory.created[0].records), 2)
            sim.close_persistent_io()


class TestResumeTargets(unittest.TestCase):
    def test_adsorbate_cmc_resume_uses_total_target_sweeps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "ads_cmc.pkl")
            thermo = str(Path(tmpdir) / "ads_cmc.dat")
            traj = str(Path(tmpdir) / "ads_cmc.traj")

            sim = AdsorbateCMC(
                atoms=_make_oh_surface(),
                calculator=ZeroCalculator(),
                T=300.0,
                adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
                adsorbate_anchor_index=0,
                substrate_elements=("Ti",),
                functional_elements=(),
                site_elements=("Ti",),
                site_type="atop",
                move_mode="hybrid",
                site_hop_prob=0.5,
                reorientation_prob=0.5,
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                seed=31,
            )
            sim._moves_per_sweep = lambda: 0
            sim.accepted_traj_file = None
            sim.rejected_traj_file = None
            sim.attempted_traj_file = None
            sim.run(nsweeps=3, traj_file=traj, interval=10, sample_interval=1, equilibration=2)

            resumed = AdsorbateCMC(
                atoms=_make_oh_surface(),
                calculator=ZeroCalculator(),
                T=300.0,
                adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
                adsorbate_anchor_index=0,
                substrate_elements=("Ti",),
                functional_elements=(),
                site_elements=("Ti",),
                site_type="atop",
                move_mode="hybrid",
                site_hop_prob=0.5,
                reorientation_prob=0.5,
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                resume=True,
                seed=31,
            )
            resumed._moves_per_sweep = lambda: 0
            resumed.accepted_traj_file = None
            resumed.rejected_traj_file = None
            resumed.attempted_traj_file = None
            resumed.run(nsweeps=5, traj_file=traj, interval=10, sample_interval=1, equilibration=2)

            self.assertEqual(resumed.sweep, 5)
            self.assertEqual(resumed.n_samples, 3)

    def test_adsorbate_cmc_chunk_mode_runs_additional_sweeps(self):
        sim = AdsorbateCMC(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.5,
            seed=33,
        )
        sim._moves_per_sweep = lambda: 0
        sim.sweep = 3
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "cmc.dat")
            sim.run(
                nsweeps=2,
                traj_file=str(Path(tmpdir) / "cmc.traj"),
                interval=10,
                sample_interval=1,
                equilibration=1,
                sweeps_are_total=False,
            )

        self.assertEqual(sim.sweep, 5)
        self.assertEqual(sim.n_samples, 1)

    def test_adsorbate_cmc_resume_restores_move_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "ads_cmc.pkl")
            thermo = str(Path(tmpdir) / "ads_cmc.dat")
            traj = str(Path(tmpdir) / "ads_cmc.traj")

            sim = AdsorbateCMC(
                atoms=_make_oh_surface(),
                calculator=ZeroCalculator(),
                T=300.0,
                adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
                adsorbate_anchor_index=0,
                substrate_elements=("Ti",),
                functional_elements=(),
                site_elements=("Ti",),
                site_type="atop",
                move_mode="displacement",
                checkpoint_file=checkpoint,
                thermo_file=thermo,
                seed=35,
                diagnostics_enabled=True,
            )
            sim._moves_per_sweep = lambda: 1
            sim._propose_move = lambda: sim.atoms.copy()
            sim.accepted_traj_file = None
            sim.rejected_traj_file = None
            sim.attempted_traj_file = None
            sim.run(
                nsweeps=1,
                traj_file=traj,
                interval=10,
                sample_interval=1,
                equilibration=0,
            )

            resumed = AdsorbateCMC(
                atoms=_make_oh_surface(),
                calculator=ZeroCalculator(),
                T=300.0,
                adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
                adsorbate_anchor_index=0,
                substrate_elements=("Ti",),
                functional_elements=(),
                site_elements=("Ti",),
                site_type="atop",
                move_mode="displacement",
                checkpoint_file=checkpoint,
                thermo_file=thermo,
                resume=True,
                seed=35,
                diagnostics_enabled=True,
            )

            self.assertEqual(
                resumed.move_diagnostics["attempted_by_move"]["displacement"],
                1,
            )
            self.assertEqual(
                resumed.move_diagnostics["accepted_by_move"]["displacement"],
                1,
            )

    def test_adsorbate_gcmc_resume_uses_total_target_sweeps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "ads_gcmc.pkl")
            thermo = str(Path(tmpdir) / "ads_gcmc.dat")
            traj = str(Path(tmpdir) / "ads_gcmc.traj")

            sim = AdsorbateGCMC(
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
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                seed=37,
            )
            sim.accepted_traj_file = None
            sim.rejected_traj_file = None
            sim.attempted_traj_file = None
            sim.run(
                nsweeps=3,
                traj_file=traj,
                interval=10,
                sample_interval=1,
                equilibration=2,
                max_moves=0,
            )

            resumed = AdsorbateGCMC(
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
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                resume=True,
                seed=37,
            )
            resumed.accepted_traj_file = None
            resumed.rejected_traj_file = None
            resumed.attempted_traj_file = None
            resumed.run(
                nsweeps=5,
                traj_file=traj,
                interval=10,
                sample_interval=1,
                equilibration=2,
                max_moves=0,
            )

            self.assertEqual(resumed.sweep, 5)
            self.assertEqual(resumed.n_samples, 3)

    def test_adsorbate_gcmc_chunk_mode_runs_additional_sweeps(self):
        sim = AdsorbateGCMC(
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
            seed=39,
        )
        sim.sweep = 3
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "gcmc.dat")
            sim.run(
                nsweeps=2,
                traj_file=str(Path(tmpdir) / "gcmc.traj"),
                interval=10,
                sample_interval=1,
                equilibration=1,
                max_moves=0,
                sweeps_are_total=False,
            )

        self.assertEqual(sim.sweep, 5)
        self.assertEqual(sim.n_samples, 1)

    def test_alloy_cmc_resume_uses_total_target_sweeps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "alloy.pkl")
            thermo = str(Path(tmpdir) / "alloy.dat")
            traj = str(Path(tmpdir) / "alloy.traj")

            sim = AlloyCMC(
                atoms=_make_ti_zr_alloy(),
                calculator=ZeroCalculator(),
                T=300.0,
                swap_elements=["Ti", "Zr"],
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                seed=41,
            )
            sim.swap_indices = []
            sim.run(nsweeps=3, traj_file=traj, interval=10, sample_interval=1, equilibration=2)

            resumed = AlloyCMC(
                atoms=_make_ti_zr_alloy(),
                calculator=ZeroCalculator(),
                T=300.0,
                swap_elements=["Ti", "Zr"],
                checkpoint_file=checkpoint,
                checkpoint_interval=100,
                thermo_file=thermo,
                resume=True,
                seed=41,
            )
            resumed.swap_indices = []
            resumed.run(nsweeps=5, traj_file=traj, interval=10, sample_interval=1, equilibration=2)

            self.assertEqual(resumed.sweep, 5)
            self.assertEqual(resumed.n_samples, 3)

    def test_alloy_cmc_chunk_mode_runs_additional_sweeps(self):
        sim = AlloyCMC(
            atoms=_make_ti_zr_alloy(),
            calculator=ZeroCalculator(),
            T=300.0,
            swap_elements=["Ti", "Zr"],
            seed=43,
        )
        sim.swap_indices = []
        sim.sweep = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "alloy.dat")
            sim.run(
                nsweeps=2,
                traj_file=str(Path(tmpdir) / "alloy.traj"),
                interval=10,
                sample_interval=1,
                equilibration=1,
                sweeps_are_total=False,
            )

        self.assertEqual(sim.sweep, 5)
        self.assertEqual(sim.n_samples, 1)


class TestMolecularIntegrity(unittest.TestCase):
    def _make_cmc_sim(self, **overrides) -> AdsorbateCMC:
        data = dict(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.5,
            seed=43,
        )
        data.update(overrides)
        return AdsorbateCMC(**data)

    def _make_gcmc_sim(self, **overrides) -> AdsorbateGCMC:
        data = dict(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            mu=-1.0,
            T=300.0,
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            max_n_adsorbates=1,
            site_elements=("Ti",),
            substrate_elements=("Ti",),
            functional_elements=(),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.5,
            seed=47,
        )
        data.update(overrides)
        return AdsorbateGCMC(**data)

    def test_detects_dissociated_molecular_adsorbate(self):
        sim = self._make_cmc_sim()
        trial = sim.atoms.copy()
        trial.positions[2] = np.array([0.0, 0.0, 4.5], dtype=float)

        self.assertFalse(sim._molecular_adsorbates_are_intact(trial))
        self.assertTrue(sim._molecular_adsorbates_are_intact(sim.atoms))

    def test_run_rejects_dissociated_starting_state(self):
        sim = self._make_cmc_sim()
        sim.atoms.positions[2] = np.array([0.0, 0.0, 4.5], dtype=float)
        sim._moves_per_sweep = lambda: 0
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "cmc.dat")
            with self.assertRaisesRegex(RuntimeError, "template bond graph"):
                sim.run(
                    nsweeps=1,
                    traj_file=str(Path(tmpdir) / "cmc.traj"),
                    interval=10,
                    sample_interval=1,
                    equilibration=0,
                )

    def test_cmc_md_rejects_dissociated_trial(self):
        sim = self._make_cmc_sim(
            enable_hybrid_md=True,
            md_move_prob=1.0,
            md_steps=1,
        )
        sim._moves_per_sweep = lambda: 1
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None
        original_positions = sim.atoms.positions.copy()

        dissociated = sim.atoms.copy()
        dissociated.positions[2] = np.array([0.0, 0.0, 4.5], dtype=float)
        sim._propose_md_move = lambda: (dissociated, -1.0, -1.0)
        sim._metropolis_accept = lambda *args, **kwargs: True

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "cmc.dat")
            sim.run(
                nsweeps=1,
                traj_file=str(Path(tmpdir) / "cmc.traj"),
                interval=10,
                sample_interval=1,
                equilibration=0,
            )

        self.assertEqual(sim.accepted_moves, 0)
        self.assertEqual(sim.md_accepted_moves, 0)
        self.assertTrue(np.allclose(sim.atoms.positions, original_positions))

    def test_cmc_md_rejects_non_upright_trial(self):
        sim = self._make_cmc_sim(
            enable_hybrid_md=True,
            md_move_prob=1.0,
            md_steps=1,
            molecular_upright_atom_indices=[1],
            molecular_upright_min_z_A=0.0,
        )
        sim._moves_per_sweep = lambda: 1
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None
        original_positions = sim.atoms.positions.copy()

        flipped = sim.atoms.copy()
        group = np.asarray(sim.ads_groups[0], dtype=int)
        flipped.positions[group[1], 2] = flipped.positions[group[0], 2] - 0.98
        sim._propose_md_move = lambda: (flipped, -1.0, -1.0)
        sim._metropolis_accept = lambda *args, **kwargs: True

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "cmc.dat")
            sim.run(
                nsweeps=1,
                traj_file=str(Path(tmpdir) / "cmc.traj"),
                interval=10,
                sample_interval=1,
                equilibration=0,
            )

        self.assertEqual(sim.accepted_moves, 0)
        self.assertEqual(sim.md_accepted_moves, 0)
        self.assertTrue(np.allclose(sim.atoms.positions, original_positions))

    def test_gcmc_md_rejects_dissociated_trial(self):
        sim = self._make_gcmc_sim(
            enable_hybrid_md=True,
            md_move_prob=1.0,
            md_steps=1,
        )
        sim.accepted_traj_file = None
        sim.rejected_traj_file = None
        sim.attempted_traj_file = None
        original_positions = sim.atoms.positions.copy()

        dissociated = sim.atoms.copy()
        dissociated.positions[2] = np.array([0.0, 0.0, 4.5], dtype=float)
        sim._propose_md_move = lambda: (dissociated, -1.0, -1.0)
        sim._metropolis_accept = lambda *args, **kwargs: True

        with tempfile.TemporaryDirectory() as tmpdir:
            sim.thermo_file = str(Path(tmpdir) / "gcmc.dat")
            sim.run(
                nsweeps=1,
                traj_file=str(Path(tmpdir) / "gcmc.traj"),
                interval=10,
                sample_interval=1,
                equilibration=0,
                max_moves=1,
            )

        self.assertEqual(sim.accepted_moves, 0)
        self.assertEqual(sim.md_accepted_moves, 0)
        self.assertTrue(np.allclose(sim.atoms.positions, original_positions))


class TestAdsorbateCMCReorientation(unittest.TestCase):
    def _make_sim(self) -> AdsorbateCMC:
        return AdsorbateCMC(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="reorientation",
            rotation_max_angle_deg=180.0,
            max_reorientation_trials=1,
            min_clearance=1.0,
            termination_clearance=0.0,
            vertical_adjust_step=0.25,
            max_vertical_adjust=1.0,
            seed=13,
        )

    def test_rotate_group_about_anchor_preserves_anchor_and_bond(self):
        sim = self._make_sim()
        group = np.asarray(sim.ads_groups[0], dtype=int)
        anchor_idx = sim.ads_anchor_indices[0]
        rotated = sim._rotate_group_about_anchor(
            group,
            axis=np.array([1.0, 0.0, 0.0]),
            angle=0.5 * np.pi,
        )

        self.assertTrue(np.allclose(rotated[0], sim.atoms.positions[anchor_idx]))
        old_bond = np.linalg.norm(
            sim.atoms.positions[group[1]] - sim.atoms.positions[group[0]]
        )
        new_bond = np.linalg.norm(rotated[1] - rotated[0])
        self.assertAlmostEqual(old_bond, new_bond, places=10)
        self.assertFalse(np.allclose(rotated[1], sim.atoms.positions[group[1]]))

    def test_reorientation_accepts_valid_rotation_without_moving_anchor(self):
        sim = self._make_sim()
        sim.rng = StubRNG(axis=[1.0, 0.0, 0.0], angle=0.5 * np.pi)
        trial = sim._propose_reorientation()

        self.assertIsNotNone(trial)
        group = np.asarray(sim.ads_groups[0], dtype=int)
        anchor_idx = sim.ads_anchor_indices[0]
        self.assertTrue(
            np.allclose(trial.positions[anchor_idx], sim.atoms.positions[anchor_idx])
        )
        old_bond = np.linalg.norm(
            sim.atoms.positions[group[1]] - sim.atoms.positions[group[0]]
        )
        new_bond = np.linalg.norm(
            trial.positions[group[1]] - trial.positions[group[0]]
        )
        self.assertAlmostEqual(old_bond, new_bond, places=10)
        self.assertFalse(
            np.allclose(trial.positions[group[1]], sim.atoms.positions[group[1]])
        )

    def test_reorientation_rejects_invalid_rotation_instead_of_lifting_anchor(self):
        sim = self._make_sim()
        sim.rng = StubRNG(axis=[1.0, 0.0, 0.0], angle=np.pi)
        trial = sim._propose_reorientation()
        self.assertIsNone(trial)

    def test_reorientation_uses_template_not_current_flipped_geometry(self):
        atoms = _make_oh_surface()
        atoms.positions[2] = np.array([0.0, 0.0, 0.82], dtype=float)
        sim = AdsorbateCMC(
            atoms=atoms,
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="reorientation",
            rotation_max_angle_deg=180.0,
            max_reorientation_trials=1,
            min_clearance=0.7,
            termination_clearance=0.0,
            seed=17,
        )
        sim.rng = StubRNG(axis=[0.0, 0.0, 1.0], angle=0.5 * np.pi)

        trial = sim._propose_reorientation()

        self.assertIsNotNone(trial)
        group = np.asarray(sim.ads_groups[0], dtype=int)
        relative = trial.positions[group[1]] - trial.positions[group[0]]
        self.assertAlmostEqual(relative[2], 0.98, places=10)

    def test_upright_filter_rejects_h_down_reorientation(self):
        sim = self._make_sim()
        sim.molecular_upright_atom_indices = (1,)
        sim.molecular_upright_min_z_A = 0.0
        sim.rng = StubRNG(axis=[1.0, 0.0, 0.0], angle=np.pi)

        trial = sim._propose_reorientation()

        self.assertIsNone(trial)

    def test_group_relative_positions_use_mic_for_wrapped_molecule(self):
        sim = AdsorbateCMC(
            atoms=_make_wrapped_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="reorientation",
            rotation_max_angle_deg=180.0,
            max_reorientation_trials=1,
            min_clearance=0.1,
            termination_clearance=0.0,
            seed=19,
        )

        group = np.asarray(sim.ads_groups[0], dtype=int)
        relative = sim._current_group_relative_positions(group)

        self.assertTrue(np.allclose(relative[0], np.zeros(3)))
        self.assertAlmostEqual(np.linalg.norm(relative[1]), 0.2, places=10)
        self.assertAlmostEqual(abs(float(relative[1, 0])), 0.2, places=10)

    def test_rotate_group_about_anchor_uses_template_for_wrapped_molecule(self):
        sim = AdsorbateCMC(
            atoms=_make_wrapped_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="reorientation",
            rotation_max_angle_deg=180.0,
            max_reorientation_trials=1,
            min_clearance=0.1,
            termination_clearance=0.0,
            seed=23,
        )

        group = np.asarray(sim.ads_groups[0], dtype=int)
        rotated = sim._rotate_group_about_anchor(
            group,
            axis=np.array([0.0, 0.0, 1.0]),
            angle=0.5 * np.pi,
        )

        self.assertTrue(np.allclose(rotated[0], sim.atoms.positions[group[0]]))
        self.assertAlmostEqual(
            np.linalg.norm(rotated[1] - rotated[0]),
            0.98,
            places=10,
        )
        self.assertGreater(np.linalg.norm(rotated[1] - rotated[0]), 0.9)


class TestAdsorbateCMCPuckering(unittest.TestCase):
    def _make_sim(self, **overrides) -> AdsorbateCMC:
        data = dict(
            atoms=_make_oh_surface(),
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="puckering",
            puckering_height_A=0.2,
            puckering_height_jitter_A=0.0,
            max_puckering_trials=4,
            min_clearance=1.0,
            termination_clearance=0.0,
            seed=29,
        )
        data.update(overrides)
        return AdsorbateCMC(**data)

    def test_puckering_lifts_support_atom_and_adsorbate_group_together(self):
        sim = self._make_sim()
        original = sim.atoms.positions.copy()

        trial = sim._propose_puckering()

        self.assertIsNotNone(trial)
        group = np.asarray(sim.ads_groups[0], dtype=int)
        support_idx = 0
        support_dz = trial.positions[support_idx, 2] - original[support_idx, 2]

        self.assertGreater(support_dz, 0.0)
        self.assertAlmostEqual(support_dz, sim.puckering_height_A)
        self.assertTrue(np.allclose(trial.positions[group, :2], original[group, :2]))
        self.assertTrue(
            np.allclose(trial.positions[group, 2] - original[group, 2], support_dz)
        )
        self.assertAlmostEqual(
            trial.positions[group[0], 2] - trial.positions[support_idx, 2],
            sim.vertical_offset,
            places=10,
        )

    def test_puckering_reseats_far_atop_anchor_to_vertical_offset(self):
        atoms = Atoms(
            "TiOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 3.5),
                (0.0, 0.0, 4.48),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Ti",),
            site_elements=("Ti",),
            support_xy_tol=1.2,
            z_max_support=5.0,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)
        original = sim.atoms.positions.copy()

        trial = sim._propose_puckering()

        self.assertIsNotNone(trial)
        self.assertGreater(trial.positions[0, 2] - original[0, 2], 0.0)
        self.assertLess(trial.positions[group[0], 2] - original[group[0], 2], 0.0)
        self.assertAlmostEqual(
            trial.positions[group[0], 2] - trial.positions[0, 2],
            sim.vertical_offset,
            places=10,
        )

    def test_puckering_reseats_laterally_offset_anchor_to_support_atom(self):
        atoms = Atoms(
            "TiOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.8, 0.0, 1.8),
                (0.8, 0.0, 2.78),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Ti",),
            site_elements=("Ti",),
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        trial = sim._propose_puckering()

        self.assertIsNotNone(trial)
        self.assertTrue(
            np.allclose(trial.positions[group[0], :2], trial.positions[0, :2])
        )
        self.assertAlmostEqual(
            trial.positions[group[0], 2] - trial.positions[0, 2],
            sim.vertical_offset,
            places=10,
        )

    def test_hybrid_probabilities_include_puckering_probability(self):
        with self.assertRaisesRegex(ValueError, "puckering_prob"):
            self._make_sim(
                move_mode="hybrid",
                site_hop_prob=0.6,
                reorientation_prob=0.3,
                puckering_prob=0.2,
            )

    def test_puckering_elements_restrict_support_atom_selection(self):
        atoms = Atoms(
            "OPtOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.4, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt", "O"),
            site_elements=("Pt", "O"),
            puckering_elements=("Pt",),
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        self.assertEqual(sim._nearest_support_atom_for_anchor(group), 1)

    def test_puckering_requires_atop_anchor(self):
        atoms = Atoms(
            "PtOH",
            positions=[
                (1.6, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[False, False, False],
        )
        atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            puckering_elements=("Pt",),
            support_xy_tol=1.2,
            termination_site_xy_tol=2.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        self.assertIsNone(sim._nearest_support_atom_for_anchor(group))
        self.assertIsNone(sim._propose_puckering())

    def test_hop_puckering_transfers_pucker_to_another_site(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_puckering",
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)
        sim.atoms.positions[0, :2] += 0.4
        sim.atoms.positions[0, 2] += 0.4
        sim.atoms.positions[group, 2] += 0.4
        original = sim.atoms.positions.copy()

        trial = sim._propose_hop_puckering()

        self.assertIsNotNone(trial)
        dz0 = trial.positions[0, 2] - original[0, 2]
        dz1 = trial.positions[1, 2] - original[1, 2]
        anchor_idx = sim.ads_anchor_indices[0]

        self.assertLess(dz0, 0.0)
        self.assertGreater(dz1, 0.0)
        self.assertTrue(
            np.allclose(
                trial.positions[0, :2],
                sim._puckering_reference_positions[0, :2],
            )
        )
        self.assertAlmostEqual(
            trial.positions[0, 2],
            sim._puckering_reference_positions[0, 2],
            places=10,
        )
        self.assertAlmostEqual(dz1, sim.puckering_height_A)
        self.assertAlmostEqual(trial.positions[anchor_idx, 0], 3.0, places=10)
        self.assertAlmostEqual(
            trial.positions[anchor_idx, 2] - trial.positions[1, 2],
            sim.vertical_offset,
            places=10,
        )

    def test_hop_reorientation_hops_anchor_and_rotates_group(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_reorientation",
            support_xy_tol=1.2,
            min_clearance=0.7,
            hop_reorientation_angle_deg=180.0,
            max_hop_reorientation_trials=1,
        )
        sim.rng = StubRNG(axis=[0.0, 1.0, 0.0], angle=0.5 * np.pi)
        group = np.asarray(sim.ads_groups[0], dtype=int)

        trial = sim._propose_hop_reorientation()

        self.assertIsNotNone(trial)
        anchor_idx = int(group[0])
        distal_idx = int(group[1])
        self.assertAlmostEqual(trial.positions[anchor_idx, 0], 3.0, places=10)
        self.assertAlmostEqual(
            trial.positions[anchor_idx, 2] - trial.positions[1, 2],
            sim.vertical_offset,
            places=10,
        )
        relative = trial.positions[distal_idx] - trial.positions[anchor_idx]
        self.assertAlmostEqual(abs(relative[0]), 0.98, places=10)
        self.assertAlmostEqual(relative[2], 0.0, places=10)

    def test_hop_reorientation_uses_template_not_current_flipped_geometry(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 0.82),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_reorientation",
            support_xy_tol=1.2,
            min_clearance=0.7,
            hop_reorientation_angle_deg=180.0,
            max_hop_reorientation_trials=1,
        )
        sim.rng = StubRNG(axis=[0.0, 0.0, 1.0], angle=0.5 * np.pi)
        group = np.asarray(sim.ads_groups[0], dtype=int)

        trial = sim._propose_hop_reorientation()

        self.assertIsNotNone(trial)
        anchor_idx = int(group[0])
        distal_idx = int(group[1])
        relative = trial.positions[distal_idx] - trial.positions[anchor_idx]
        self.assertAlmostEqual(relative[2], 0.98, places=10)

    def test_site_hop_uses_template_not_current_flipped_geometry(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 0.82),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="site_hop",
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        trial = sim._propose_site_hop()

        self.assertIsNotNone(trial)
        anchor_idx = int(group[0])
        distal_idx = int(group[1])
        relative = trial.positions[distal_idx] - trial.positions[anchor_idx]
        self.assertAlmostEqual(relative[2], 0.98, places=10)

    def test_site_hop_resets_previously_puckered_source_support(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="site_hop",
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)
        sim.atoms.positions[0, 2] += 0.4
        sim.atoms.positions[group, 2] += 0.4

        trial = sim._propose_site_hop()

        self.assertIsNotNone(trial)
        self.assertAlmostEqual(
            trial.positions[0, 2],
            sim._puckering_reference_positions[0, 2],
            places=10,
        )
        self.assertAlmostEqual(
            trial.positions[1, 2],
            sim._puckering_reference_positions[1, 2],
            places=10,
        )
        self.assertAlmostEqual(trial.positions[group[0], 0], 3.0, places=10)

    def test_hop_puckering_uses_template_not_current_flipped_geometry(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 0.82),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_puckering",
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        trial = sim._propose_hop_puckering()

        self.assertIsNotNone(trial)
        anchor_idx = int(group[0])
        distal_idx = int(group[1])
        relative = trial.positions[distal_idx] - trial.positions[anchor_idx]
        self.assertAlmostEqual(relative[2], 0.98, places=10)

    def test_hop_puckering_can_start_from_non_atop_adsorbate(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (1.5, 0.0, 1.8),
                (1.5, 0.0, 2.78),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_puckering",
            support_xy_tol=1.2,
            min_clearance=0.7,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)

        self.assertIsNone(sim._nearest_support_atom_for_anchor(group))
        trial = sim._propose_hop_puckering()

        self.assertIsNotNone(trial)
        support_dz = trial.positions[:2, 2] - sim._puckering_reference_positions[:2, 2]
        target_support = int(np.argmax(support_dz))
        anchor_idx = int(group[0])
        self.assertAlmostEqual(
            trial.positions[target_support, 2]
            - sim._puckering_reference_positions[target_support, 2],
            sim.puckering_height_A,
            places=10,
        )
        self.assertAlmostEqual(
            trial.positions[anchor_idx, 2] - trial.positions[target_support, 2],
            sim.vertical_offset,
            places=10,
        )

    def test_hop_puckering_reorientation_transfers_pucker_and_rotates_group(self):
        atoms = Atoms(
            "Pt2OH",
            positions=[
                (0.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.0, 0.0, 2.78),
            ],
            cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=[True, True, False],
        )
        atoms.set_tags([0, 0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
        sim = self._make_sim(
            atoms=atoms,
            substrate_elements=("Pt",),
            site_elements=("Pt",),
            site_type="atop",
            move_mode="hop_puckering_reorientation",
            support_xy_tol=1.2,
            min_clearance=0.7,
            hop_reorientation_angle_deg=180.0,
            max_hop_reorientation_trials=1,
        )
        sim.rng = StubRNG(axis=[0.0, 1.0, 0.0], angle=0.5 * np.pi)
        group = np.asarray(sim.ads_groups[0], dtype=int)
        sim.atoms.positions[0, 2] += 0.4
        sim.atoms.positions[group, 2] += 0.4

        trial = sim._propose_hop_puckering_reorientation()

        self.assertIsNotNone(trial)
        anchor_idx = int(group[0])
        distal_idx = int(group[1])
        self.assertAlmostEqual(
            trial.positions[0, 2],
            sim._puckering_reference_positions[0, 2],
            places=10,
        )
        self.assertAlmostEqual(
            trial.positions[1, 2] - sim._puckering_reference_positions[1, 2],
            sim.puckering_height_A,
            places=10,
        )
        self.assertAlmostEqual(trial.positions[anchor_idx, 0], 3.0, places=10)
        self.assertAlmostEqual(
            trial.positions[anchor_idx, 2] - trial.positions[1, 2],
            sim.vertical_offset,
            places=10,
        )
        relative = trial.positions[distal_idx] - trial.positions[anchor_idx]
        self.assertAlmostEqual(abs(relative[0]), 0.98, places=10)
        self.assertAlmostEqual(relative[2], 0.0, places=10)

    def test_attempted_traj_records_filter_failed_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sim = self._make_sim(
                move_mode="displacement",
                support_xy_tol=0.2,
                z_max_support=1.0,
                diagnostics_enabled=True,
            )

            invalid = sim.atoms.copy()
            group = np.asarray(sim.ads_groups[0], dtype=int)
            invalid.positions[group, :2] += 4.0
            sim._propose_move = lambda: invalid.copy()

            attempted = root / "attempted.traj"
            rejected = root / "rejected.traj"
            samples = root / "samples.traj"
            sim.attempted_traj_file = str(attempted)
            sim.rejected_traj_file = str(rejected)

            stats = sim.run(
                nsweeps=1,
                traj_file=str(samples),
                interval=1,
                sample_interval=1,
                equilibration=0,
            )

            attempted_frames = read(str(attempted), ":") if attempted.exists() else []
            rejected_frames = read(str(rejected), ":") if rejected.exists() else []
            self.assertEqual(len(attempted_frames), 1)
            self.assertEqual(len(rejected_frames), 1)
            self.assertEqual(rejected_frames[0].info.get("mc_event"), "rejected")
            self.assertEqual(
                rejected_frames[0].info.get("mc_reject_reason"),
                "afloat_adsorbate",
            )
            self.assertEqual(rejected_frames[0].info.get("mc_move_name"), "displacement")
            diagnostics = stats["move_diagnostics"]
            self.assertEqual(diagnostics["attempted_by_move"]["displacement"], 1)
            self.assertEqual(diagnostics["rejected_by_move"]["displacement"], 1)
            self.assertEqual(diagnostics["rejected_by_reason"]["afloat_adsorbate"], 1)

    def test_move_diagnostics_are_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sim = self._make_sim(move_mode="displacement")
            sim._propose_move = lambda: sim.atoms.copy()

            stats = sim.run(
                nsweeps=1,
                traj_file=str(root / "samples.traj"),
                interval=1,
                sample_interval=1,
                equilibration=0,
            )

            self.assertEqual(stats["move_diagnostics"]["attempted_by_move"], {})
            self.assertEqual(sim.move_diagnostics["attempted_by_move"], {})

    def test_debug_traj_interval_thins_attempted_and_accepted_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sim = self._make_sim(
                move_mode="displacement",
                debug_traj_interval=2,
            )
            sim._propose_move = lambda: sim.atoms.copy()

            attempted = root / "attempted.traj"
            accepted = root / "accepted.traj"
            samples = root / "samples.traj"
            sim.attempted_traj_file = str(attempted)
            sim.accepted_traj_file = str(accepted)

            sim.run(
                nsweeps=3,
                traj_file=str(samples),
                interval=1,
                sample_interval=1,
                equilibration=0,
            )

            attempted_frames = read(str(attempted), ":") if attempted.exists() else []
            accepted_frames = read(str(accepted), ":") if accepted.exists() else []
            self.assertEqual(len(attempted_frames), 1)
            self.assertEqual(len(accepted_frames), 1)

    def test_surface_side_filter_rejects_buried_molecular_atom(self):
        atoms = Atoms(
            "TiOOH",
            positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.8),
                (0.9, 0.0, -0.1),
                (1.7, 0.0, -0.1),
            ],
            cell=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 3.0]],
            pbc=[False, False, True],
        )
        atoms.set_tags(
            [0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET]
        )
        sim = AdsorbateCMC(
            atoms=atoms,
            calculator=ZeroCalculator(),
            T=300.0,
            adsorbate_element="O",
            adsorbate=Atoms(
                "OOH",
                positions=[(0.0, 0.0, 0.0), (0.9, 0.0, -1.9), (1.7, 0.0, -1.9)],
            ),
            adsorbate_anchor_index=0,
            substrate_elements=("Ti",),
            functional_elements=(),
            site_elements=("Ti",),
            site_type="atop",
            move_mode="reorientation",
            min_clearance=0.7,
            support_xy_tol=1.2,
            adsorbate_surface_clearance_A=0.0,
            adsorbate_surface_xy_tol_A=1.2,
            seed=31,
        )
        group = np.asarray(sim.ads_groups[0], dtype=int)
        buried = atoms.positions[group].copy()
        lifted = buried.copy()
        lifted[1:, 2] = 0.2

        self.assertFalse(sim._group_positions_are_valid(group, buried, atoms=atoms))
        self.assertTrue(sim._group_positions_are_valid(group, lifted, atoms=atoms))


class TestAmbiguousEmptyMolecularAdsorbates(unittest.TestCase):
    def test_clean_trial_without_tags_is_treated_as_empty(self):
        slab = Atoms(
            symbols=["Ti", "O"],
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.8)],
            cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 15.0]],
            pbc=[False, False, False],
        )
        sim = AdsorbateGCMC(
            atoms=slab,
            calculator=ZeroCalculator(),
            mu=-1.0,
            T=300.0,
            adsorbate=Atoms("OH", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)]),
            adsorbate_anchor_index=0,
            site_elements=("Ti",),
            substrate_elements=("Ti",),
            functional_elements=("O",),
            site_type="atop",
            move_mode="hybrid",
            site_hop_prob=0.5,
            reorientation_prob=0.25,
            enable_hybrid_md=True,
            md_without_adsorbate=True,
            allow_ambiguous_empty_adsorbates=True,
            seed=17,
        )

        trial = sim.atoms.copy()
        self.assertEqual(sim._adsorbate_groups_for_atoms(trial), [])
        self.assertFalse(sim.has_afloat_adsorbates(trial))


if __name__ == "__main__":
    unittest.main()
