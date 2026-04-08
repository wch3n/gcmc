import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian

from gcmc.workflows import (
    AdsorbateGCMCScanWorkflow,
    _prepare_adsorbate_scan_atoms,
    _prepare_alloy_atoms,
)


def _make_two_sided_mxene_like_slab() -> Atoms:
    symbols = ["Ti", "Zr", "C", "C", "Mo", "Ti", "O", "O"]
    positions = [
        (0.0, 0.0, 2.0),
        (2.0, 0.0, 2.1),
        (0.0, 0.0, 5.0),
        (2.0, 0.0, 5.0),
        (0.0, 0.0, 8.0),
        (2.0, 0.0, 8.1),
        (0.0, 0.0, 9.2),
        (2.0, 0.0, 0.9),
    ]
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
        pbc=[True, True, False],
    )


def _constraint_of_type(atoms: Atoms, cls):
    for constraint in atoms.constraints:
        if isinstance(constraint, cls):
            return constraint
    return None


class TestWorkflowLayerZConstraints(unittest.TestCase):
    def test_prepare_alloy_atoms_fix_z_by_layer_index(self):
        atoms = _make_two_sided_mxene_like_slab()
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            fix_below_z=None,
            fix_z_elements=["Ti", "Zr", "Mo"],
            fix_z_layers={"top": [1], "bottom": [1]},
            surface_layer_tol=0.5,
        )

        prepared = _prepare_alloy_atoms(atoms, cfg)
        constraint = _constraint_of_type(prepared, FixCartesian)

        self.assertIsNotNone(constraint)
        self.assertTrue(np.array_equal(np.sort(constraint.get_indices()), np.array([0, 1, 4, 5])))
        self.assertTrue(np.array_equal(np.asarray(constraint.mask, dtype=bool), np.array([False, False, True])))

    def test_prepare_adsorbate_scan_atoms_fix_z_bottom_only(self):
        atoms = _make_two_sided_mxene_like_slab()
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            fix_below_z=None,
            fix_z_elements=["Ti", "Zr", "Mo"],
            fix_z_layers={"bottom": [1]},
            surface_layer_tol=0.5,
        )

        prepared = _prepare_adsorbate_scan_atoms(atoms, cfg, {})
        constraint = _constraint_of_type(prepared, FixCartesian)

        self.assertIsNotNone(constraint)
        self.assertTrue(np.array_equal(np.sort(constraint.get_indices()), np.array([0, 1])))

    def test_fix_below_z_and_fix_z_layers_can_coexist(self):
        atoms = _make_two_sided_mxene_like_slab()
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            fix_below_z=1.0,
            fix_z_elements=["Ti", "Zr", "Mo"],
            fix_z_layers={"top": [1]},
            surface_layer_tol=0.5,
        )

        prepared = _prepare_alloy_atoms(atoms, cfg)
        cartesian = _constraint_of_type(prepared, FixCartesian)
        fixed = _constraint_of_type(prepared, FixAtoms)

        self.assertIsNotNone(cartesian)
        self.assertIsNotNone(fixed)
        self.assertTrue(np.array_equal(np.sort(cartesian.get_indices()), np.array([4, 5])))
        self.assertTrue(np.array_equal(np.sort(fixed.get_indices()), np.array([7])))


class _FakeRayModule:
    def __init__(self):
        self._initialized = False
        self.init_calls = 0
        self.shutdown_calls = 0

    def is_initialized(self):
        return self._initialized

    def init(self, **kwargs):
        self._initialized = True
        self.init_calls += 1

    def shutdown(self):
        self._initialized = False
        self.shutdown_calls += 1


class TestMuExchangeSeedParallelism(unittest.TestCase):
    def _make_workflow(self, **overrides):
        data = {
            "seeds": [67, 68, 69],
            "mu_values": [-1.0, -0.9],
            "mu_exchange_parallel_seeds": True,
            "mu_exchange_max_concurrent_seeds": None,
            "swap_interval": 20,
            "nsweeps": 100,
            "equilibration": 40,
            "mu_exchange_cycles": None,
            "mu_exchange_equilibration_cycles": None,
            "ray_log_to_driver": False,
            "ray_address": None,
        }
        data.update(overrides)
        config = SimpleNamespace(**data)
        return AdsorbateGCMCScanWorkflow(config, calculator_factory=lambda *_: None)

    def test_parallel_seed_ladders_use_threads_for_ray_backend(self):
        workflow = self._make_workflow()
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run(out_dir, *, seed, backend):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return [{"mu": -1.0, "seed": seed}]

        fake_ray = _FakeRayModule()
        with mock.patch.object(workflow, "_run_mu_exchange_seed", side_effect=fake_run):
            with mock.patch.dict(sys.modules, {"ray": fake_ray}):
                results = workflow._run_mu_exchange_scan(Path("/tmp"), backend="ray")

        self.assertEqual(sorted(result["seed"] for result in results), [67, 68, 69])
        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(fake_ray.init_calls, 1)
        self.assertEqual(fake_ray.shutdown_calls, 1)

    def test_parallel_seed_ladders_reject_multiprocessing_backend(self):
        workflow = self._make_workflow()
        with self.assertRaisesRegex(
            ValueError, "mu_exchange_parallel_seeds currently requires backend='ray'"
        ):
            workflow._run_mu_exchange_scan(Path("/tmp"), backend="multiprocessing")

    def test_mu_exchange_cycles_override_nsweeps(self):
        workflow = self._make_workflow(
            mu_exchange_cycles=7,
            mu_exchange_equilibration_cycles=2,
            nsweeps=999,
            equilibration=123,
            swap_interval=15,
        )
        total_sweeps, equilibration_sweeps = workflow._resolve_mu_exchange_schedule(
            workflow.config
        )
        self.assertEqual(total_sweeps, 105)
        self.assertEqual(equilibration_sweeps, 30)

    def test_mu_exchange_schedule_falls_back_to_nsweeps(self):
        workflow = self._make_workflow(
            mu_exchange_cycles=None,
            nsweeps=240,
            equilibration=60,
            swap_interval=20,
        )
        total_sweeps, equilibration_sweeps = workflow._resolve_mu_exchange_schedule(
            workflow.config
        )
        self.assertEqual(total_sweeps, 240)
        self.assertEqual(equilibration_sweeps, 60)


if __name__ == "__main__":
    unittest.main()
