from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from tools.active_learning.evaluation import (
    build_parser as evaluation_parser,
    centered_error_values,
    iter_structure_dirs,
    plot_committee_diagnostics,
    plot_learning_curve,
    relative_energy_plot_data,
)
from tools.active_learning.models import discover_model_paths, resolve_models
from tools.active_learning.selection import (
    build_parser as selection_parser,
    load_trajectory_frames,
    priority_aware_maxmin,
    resolve_input_trajectories,
    resolve_reference_pools,
    trajectory_provenance,
)


class ActiveLearningModelTests(unittest.TestCase):
    def test_discovers_native_committee_in_stable_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            second = root / "committee-01.model"
            first = root / "committee-00.model"
            second.touch()
            first.touch()

            self.assertEqual(
                discover_model_paths([str(root)], "auto"),
                [first.resolve(), second.resolve()],
            )

    def test_auto_rejects_mixed_model_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "committee.model").touch()
            (root / "committee.json").touch()

            with self.assertRaisesRegex(ValueError, "found both"):
                discover_model_paths([str(root)], "auto")

    def test_selection_requires_two_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "committee.model"
            model.touch()

            with self.assertRaisesRegex(ValueError, "At least 2"):
                resolve_models([str(model)], "auto", minimum=2)

    def test_reference_pools_are_inferred_beside_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "committee.model"
            train = root / "dataset_train.xyz"
            test = root / "dataset_test.xyz"
            model.touch()
            train.touch()
            test.touch()

            model_pools, dft_pools = resolve_reference_pools(
                [model],
                model_seen_values=None,
                dft_seen_values=None,
                legacy_values=None,
                no_reference_pool=False,
            )

            self.assertEqual(model_pools, [train.resolve()])
            self.assertEqual(dft_pools, [test.resolve(), train.resolve()])


class ActiveLearningParserTests(unittest.TestCase):
    def test_selection_defaults(self):
        args = selection_parser().parse_args(["--model-path", "committee"])

        self.assertEqual(args.frame_stride, 2)
        self.assertEqual(args.candidate_pool_size, 300)
        self.assertEqual(args.soap_r_cut, 6.0)
        self.assertEqual(args.soap_n_max, 8)
        self.assertEqual(args.soap_l_max, 6)
        self.assertEqual(args.novelty_weight, 0.45)
        self.assertEqual(args.model_path, ["committee"])
        self.assertIsNone(args.temperatures)
        self.assertEqual(args.input_tail_fraction, 1.0)
        self.assertEqual(args.max_frames_per_trajectory, 0)
        self.assertIsNone(args.trajectory_kinds)
        self.assertFalse(args.include_unevaluated_rejected)
        self.assertEqual(args.min_per_trajectory_kind, 0)
        self.assertEqual(args.min_per_temperature, 0)

    def test_evaluation_uses_same_model_path_interface(self):
        args = evaluation_parser().parse_args(["--model-path", "committee"])

        self.assertEqual(args.model_path, ["committee"])
        self.assertEqual(args.calculator, "auto")
        self.assertEqual(args.plot_output, "mace_committee_diagnostics")
        self.assertEqual(args.learning_curve_plot, "mace_learning_curve")
        self.assertFalse(args.no_plots)


class ActiveLearningInputDiscoveryTests(unittest.TestCase):
    @staticmethod
    def write_trajectory(
        path: Path,
        n_frames: int = 2,
        energy_indices: set[int] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Trajectory(str(path), "w") as writer:
            for index in range(n_frames):
                atoms = Atoms(
                    "HO",
                    positions=[[0.0, 0.0, float(index)], [0.0, 0.0, 1.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                )
                if energy_indices is not None and index in energy_indices:
                    atoms.info["mc_energy_eV"] = float(index)
                writer.write(atoms)

    def test_campaign_discovery_merges_snapshots_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "oh"
            expected = [
                root
                / "runs"
                / "snapshot_000"
                / "seed_067"
                / "results"
                / "replica_300K.traj",
                root
                / "runs"
                / "snapshot_001"
                / "seed_068"
                / "results"
                / "replica_300K.traj",
            ]
            excluded_temperature = (
                root
                / "runs"
                / "snapshot_000"
                / "seed_067"
                / "results"
                / "replica_800K.traj"
            )
            excluded_debug = expected[0].with_name(
                "replica_300K_attempted.traj"
            )
            expected.extend(
                [
                    expected[0].with_name("replica_300K_accepted.traj"),
                    expected[0].with_name("replica_300K_rejected.traj"),
                ]
            )
            for path in [*expected, excluded_temperature, excluded_debug]:
                self.write_trajectory(path)

            discovered = resolve_input_trajectories(
                [str(root)], temperatures={300.0}
            )

        self.assertEqual(
            discovered,
            sorted((path.resolve() for path in expected), key=str),
        )

    def test_campaign_provenance_is_extracted_from_path(self):
        path = Path(
            "/campaign/runs/snapshot_012/seed_067/results/replica_539K.traj"
        )

        self.assertEqual(
            trajectory_provenance(path),
            {
                "source_snapshot": "snapshot_012",
                "source_seed": 67,
                "source_temperature_K": 539.0,
                "source_trajectory_kind": "sampled",
            },
        )

    def test_tail_and_per_trajectory_cap_preserve_source_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replica_300K.traj"
            self.write_trajectory(path, n_frames=10)

            frames, indices = load_trajectory_frames(
                path,
                frame_stride=2,
                tail_fraction=0.5,
                max_frames=2,
            )

        self.assertEqual(len(frames), 2)
        self.assertEqual(indices, [5, 9])

    def test_rejected_pool_can_require_energy_evaluated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replica_300K_rejected.traj"
            self.write_trajectory(path, n_frames=6, energy_indices={0, 2, 4})

            frames, indices = load_trajectory_frames(
                path,
                frame_stride=1,
                max_frames=2,
                require_mc_energy=True,
            )

        self.assertEqual(len(frames), 2)
        self.assertEqual(indices, [0, 4])

    def test_maxmin_selection_preserves_required_group_seeds(self):
        descriptors = np.asarray([[0.0], [0.1], [1.0], [2.0]])
        priorities = np.asarray([1.0, 0.8, 0.2, 0.1])

        selected = priority_aware_maxmin(
            descriptors,
            priorities,
            n_select=3,
            initial_indices=[2],
        )

        self.assertEqual(selected[0], 2)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)


class ActiveLearningEvaluationDiscoveryTests(unittest.TestCase):
    def test_recursive_discovery_includes_only_active_learning_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = [
                root / "al_000" / "test" / "000",
                root / "al_001" / "test" / "001",
            ]
            excluded = [
                root / "al_000" / "train" / "000",
                root / "al_001" / "train" / "001",
                root / "other" / "test" / "000",
                root / "legacy" / "001",
            ]
            for directory in [*expected, *excluded]:
                directory.mkdir(parents=True)
                (directory / "POSCAR").touch()

            entries = iter_structure_dirs(
                root,
                labels=["000", "001"],
                recursive=True,
            )

        self.assertEqual(
            [(relative, series) for relative, series, _, _ in entries],
            [
                ("al_000/test/000", "al_000"),
                ("al_001/test/001", "al_001"),
            ],
        )

    def test_recursive_discovery_accepts_a_test_directory_as_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_root = Path(tmp) / "al_003" / "test"
            calculation = test_root / "000"
            calculation.mkdir(parents=True)
            (calculation / "POSCAR").touch()

            entries = iter_structure_dirs(
                test_root,
                labels=["000"],
                recursive=True,
            )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0], "000")
        self.assertEqual(entries[0][1], "al_003")


class ActiveLearningEvaluationPlotTests(unittest.TestCase):
    def test_relative_energies_are_centered_by_composition(self):
        records = [
            {
                "formula": "X2",
                "n_atoms": 2,
                "_dft_energy": 0.0,
                "_ensemble_energy_error_per_atom": 0.01,
            },
            {
                "formula": "X2",
                "n_atoms": 2,
                "_dft_energy": 0.04,
                "_ensemble_energy_error_per_atom": 0.03,
            },
        ]
        energies = np.asarray([[0.01, 0.02], [0.07, 0.08]])

        indices, dft_relative, model_relative = relative_energy_plot_data(
            records, energies
        )

        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(dft_relative, [-10.0, 10.0])
        np.testing.assert_allclose(
            model_relative,
            [[-15.0, -15.0], [15.0, 15.0]],
        )
        np.testing.assert_allclose(
            centered_error_values(
                records, "_ensemble_energy_error_per_atom"
            ),
            [-0.01, 0.01],
        )

    def test_diagnostic_plots_are_written_as_pdf_and_png(self):
        records = []
        dft_energies = np.asarray([-2.0, -1.9, -1.8, -1.7])
        energies = np.column_stack(
            (dft_energies + 0.02, dft_energies - 0.01)
        )
        dft_forces = []
        forces_by_model = [[], []]
        for index, dft_energy in enumerate(dft_energies):
            reference = np.asarray(
                [[0.1 + index * 0.01, 0.0, -0.1], [0.0, 0.2, -0.2]]
            )
            first = reference + 0.02
            second = reference - 0.01
            dft_forces.append(reference)
            forces_by_model[0].append(first)
            forces_by_model[1].append(second)
            records.append(
                {
                    "directory": f"al_{index // 2:03d}/test/{index:03d}",
                    "series": f"al_{index // 2:03d}",
                    "formula": "X2",
                    "n_atoms": 2,
                    "_dft_energy": dft_energy,
                    "ensemble_energy_std_meV_atom": 7.5,
                    "committee_force_std_rms_eV_A": 0.025,
                    "ensemble_force_rmse_eV_A": 0.005,
                }
            )
        learning_rows = [
            {
                "scope": scope,
                "series": series,
                "ensemble_relative_energy_rmse_meV_atom": energy_rmse,
                "ensemble_force_rmse_eV_A": force_rmse,
            }
            for series, energy_rmse, force_rmse in (
                ("al_000", 12.0, 0.08),
                ("al_001", 8.0, 0.05),
            )
            for scope in ("round", "cumulative")
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diagnostic_paths = plot_committee_diagnostics(
                records,
                energies,
                forces_by_model,
                dft_forces,
                ["model_00", "model_01"],
                root / "diagnostics",
            )
            learning_paths = plot_learning_curve(
                learning_rows, root / "learning"
            )

            for path in [*diagnostic_paths, *learning_paths]:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
