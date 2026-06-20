import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read

from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.utils import initialize_surface_adsorbates
from gcmc.workflows import (
    AdsorbateCMCWorkflow,
    AdsorbateGCMCScanWorkflow,
    AlloyCMCWorkflow,
    _prepare_adsorbate_scan_atoms,
    _prepare_alloy_atoms,
    _write_site_overlay_if_requested,
    load_adsorbate_cmc_config,
    load_alloy_cmc_config,
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
    def test_prepare_atoms_can_clear_loaded_constraints(self):
        atoms = _make_two_sided_mxene_like_slab()
        atoms.set_constraint(FixAtoms(indices=[0, 1]))
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            clear_loaded_constraints=True,
            fix_below_z=None,
            fix_z_elements=[],
            fix_z_layers=None,
            surface_layer_tol=0.5,
        )

        prepared = _prepare_alloy_atoms(atoms, cfg)

        self.assertEqual(len(prepared.constraints), 0)

    def test_prepare_atoms_clear_loaded_constraints_then_apply_workflow_constraints(self):
        atoms = _make_two_sided_mxene_like_slab()
        atoms.set_constraint(FixAtoms(indices=[0, 1]))
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            clear_loaded_constraints=True,
            fix_below_z=None,
            fix_z_elements=["Ti", "Zr", "Mo"],
            fix_z_layers={"top": [1]},
            surface_layer_tol=0.5,
        )

        prepared = _prepare_alloy_atoms(atoms, cfg)
        cartesian = _constraint_of_type(prepared, FixCartesian)
        fixed = _constraint_of_type(prepared, FixAtoms)

        self.assertIsNotNone(cartesian)
        self.assertIsNone(fixed)
        self.assertTrue(np.array_equal(np.sort(cartesian.get_indices()), np.array([4, 5])))

    def test_prepare_alloy_atoms_fix_z_by_layer_index(self):
        atoms = _make_two_sided_mxene_like_slab()
        cfg = SimpleNamespace(
            supercell_matrix=None,
            repeat=(1, 1, 1),
            clear_loaded_constraints=False,
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
            clear_loaded_constraints=False,
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
            clear_loaded_constraints=False,
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

    def test_adsorbate_cmc_output_dir_prefixes_relative_outputs(self):
        cfg = SimpleNamespace(
            snapshot="dummy.traj",
            frame=0,
            output_dir="/tmp/adsorbate_cmc_outputs",
            output_prefix="seed_067/out",
        )
        workflow = AdsorbateCMCWorkflow(cfg, calculator_factory=lambda *_: None)

        output_paths = workflow._build_output_paths()

        self.assertEqual(
            output_paths["traj_file"],
            "/tmp/adsorbate_cmc_outputs/seed_067/out.traj",
        )
        self.assertEqual(
            output_paths["thermo_file"],
            "/tmp/adsorbate_cmc_outputs/seed_067/out.dat",
        )
        self.assertEqual(
            output_paths["checkpoint_file"],
            "/tmp/adsorbate_cmc_outputs/seed_067/out.pkl",
        )
        self.assertEqual(
            output_paths["initial_traj_file"],
            "/tmp/adsorbate_cmc_outputs/seed_067/out_initial.traj",
        )
        self.assertEqual(
            output_paths["site_overlay_file"],
            "/tmp/adsorbate_cmc_outputs/seed_067/out_sites.traj",
        )

    def test_load_adsorbate_cmc_config_preserves_relative_output_prefix_with_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
system:
  snapshot: slab.traj
cmc:
  seed: 67
output:
  output_dir: results
  output_prefix: seed_067/out
""".strip()
            )

            cfg = load_adsorbate_cmc_config(config_path)

        self.assertEqual(cfg.output_dir, str((Path(tmpdir) / "results").resolve()))
        self.assertEqual(cfg.output_prefix, "seed_067/out")

    def test_alloy_cmc_output_dir_prefixes_relative_outputs(self):
        cfg = SimpleNamespace(
            snapshot="dummy.traj",
            frame=0,
            output_dir="/tmp/alloy_cmc_outputs",
            output_prefix="seed_067/out",
        )
        workflow = AlloyCMCWorkflow(cfg, calculator_factory=lambda *_: None)

        output_paths = workflow._build_output_paths()

        self.assertEqual(
            output_paths["traj_file"],
            "/tmp/alloy_cmc_outputs/seed_067/out.traj",
        )
        self.assertEqual(
            output_paths["accepted_traj_file"],
            "/tmp/alloy_cmc_outputs/seed_067/out_accepted.traj",
        )
        self.assertEqual(
            output_paths["thermo_file"],
            "/tmp/alloy_cmc_outputs/seed_067/out.dat",
        )
        self.assertEqual(
            output_paths["checkpoint_file"],
            "/tmp/alloy_cmc_outputs/seed_067/out.pkl",
        )

    def test_load_alloy_cmc_config_preserves_relative_output_prefix_with_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                """
system:
  snapshot: slab.traj
cmc:
  seed: 67
output:
  output_dir: results
  output_prefix: seed_067/out
""".strip()
            )

            cfg = load_alloy_cmc_config(config_path)

        self.assertEqual(cfg.output_dir, str((Path(tmpdir) / "results").resolve()))
        self.assertEqual(cfg.output_prefix, "seed_067/out")

    def test_adsorbate_cmc_multi_seed_run_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                snapshot="dummy.traj",
                frame=0,
                temperature=300.0,
                move_mode="hybrid",
                enable_hybrid_md=False,
                md_move_prob=0.0,
                md_steps=0,
                md_timestep_fs=1.0,
                md_planar=False,
                nsweeps=10,
                write_interval=2,
                sample_interval=1,
                equilibration=0,
                seed=81,
                seeds=[67, 68],
                backend="multiprocessing",
                n_workers=1,
                output_dir=tmpdir,
                output_prefix="out",
            )
            workflow = AdsorbateCMCWorkflow(cfg, calculator_factory=lambda *_: None)

            def _fake_single_seed(seed, *, multi_seed, task=None, emit_status=True):
                output_paths = workflow._build_output_paths(
                    seed=seed, multi_seed=multi_seed
                )
                return (
                    {
                        "acceptance": 10.0 + seed,
                        "energy": -1.0 * seed,
                        "cv": 0.1 * seed,
                    },
                    output_paths,
                    {},
                )

            def _fake_runner(tasks, run_one, *, n_workers, status_formatter=None):
                results = []
                for task in tasks:
                    result = run_one(task)
                    results.append(result)
                    if status_formatter is not None:
                        status_formatter(result)
                return results

            with mock.patch.object(workflow, "_run_single_seed", side_effect=_fake_single_seed):
                with mock.patch(
                    "gcmc.workflows.run_tasks_with_multiprocessing",
                    side_effect=_fake_runner,
                ):
                    results = workflow.run()

            self.assertEqual([row["seed"] for row in results], [67, 68])
            self.assertEqual(
                results[0]["traj_file"],
                str(Path(tmpdir) / "seed_067" / "out.traj"),
            )
            self.assertTrue((Path(tmpdir) / "summary.csv").is_file())

    def test_write_site_overlay_if_requested_creates_overlay_traj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            atoms = _make_two_sided_mxene_like_slab()
            cfg = SimpleNamespace(
                write_site_overlay=True,
                site_overlay_include_blocked=True,
                site_overlay_z_field="suggested_z_A",
                site_elements=["Ti", "Zr", "Mo"],
                top_layer_element=None,
                substrate_elements=["Ti", "Zr", "Mo", "C"],
                surface_side="top",
                site_type=["atop", "fcc", "hcp"],
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                bridge_cutoff=None,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.5,
                termination_clearance=0.0,
            )
            out_path = Path(tmpdir) / "sites.traj"

            _write_site_overlay_if_requested(
                atoms,
                cfg,
                output_file=out_path,
                functional_elements=("O",),
            )

            self.assertTrue(out_path.is_file())
            overlay = read(out_path)
            self.assertGreater(len(overlay), len(atoms))


class TestMolecularFixedCountInitialization(unittest.TestCase):
    def test_initialize_surface_adsorbates_tags_molecular_groups(self):
        atoms = _make_two_sided_mxene_like_slab()
        adsorbate = Atoms(
            symbols=["O", "H"],
            positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.98)],
        )

        initialized, _, _ = initialize_surface_adsorbates(
            atoms,
            adsorbate=adsorbate,
            n_adsorbates=1,
            site_elements=["Ti", "Zr", "Mo"],
            substrate_elements=["Ti", "Zr", "Mo", "C"],
            surface_side="top",
            site_types=["atop"],
            layer_tol=0.5,
            xy_tol=0.6,
            support_xy_tol=1.2,
            termination_elements=["O"],
            min_termination_dist=0.8,
            anchor_index=0,
            seed=67,
        )

        tags = np.asarray(initialized.get_tags(), dtype=int)
        added_tags = tags[len(atoms) :]

        self.assertEqual(len(added_tags), 2)
        self.assertTrue(np.all(added_tags >= ADSORBATE_TAG_OFFSET))
        self.assertEqual(len(np.unique(added_tags)), 1)


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
