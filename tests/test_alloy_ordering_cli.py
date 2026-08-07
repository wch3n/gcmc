from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from gcmc.analysis.alloy_ordering_cli import (
    _finite_random_layer_polarization,
    build_parser,
    run,
)


class AlloyOrderingCLITests(unittest.TestCase):
    def test_finite_random_layer_polarization(self) -> None:
        four_site_layers = np.asarray([0, 0, 1, 1])
        self.assertAlmostEqual(
            _finite_random_layer_polarization(four_site_layers, 2),
            1.0 / 3.0,
        )

    def _frame(self, symbols: list[str]) -> Atoms:
        positions = [
            (0.0, 0.0, 4.0),
            (1.0, 0.0, 4.0),
            (0.0, 1.0, 4.0),
            (1.0, 1.0, 4.0),
            (0.0, 0.0, 6.0),
            (1.0, 0.0, 6.0),
            (0.0, 1.0, 6.0),
            (1.0, 1.0, 6.0),
        ]
        return Atoms(symbols=symbols, positions=positions, cell=(2.0, 2.0, 10.0), pbc=True)

    def test_cli_summarizes_sro_and_layer_polarization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            results = run_dir / "results"
            results.mkdir()
            (run_dir / "config.yaml").write_text(
                """system:
  composition:
    Ti: 0.5
    Zr: 0.5
output:
  output_dir: results
""",
                encoding="utf-8",
            )

            polarized = self._frame(
                ["Ti", "Ti", "Ti", "Zr", "Zr", "Zr", "Zr", "Ti"]
            )
            checkerboard = self._frame(
                ["Ti", "Zr", "Zr", "Ti", "Ti", "Zr", "Zr", "Ti"]
            )
            write(results / "replica_300K.traj", [polarized] * 6)
            write(results / "replica_600K.traj", [checkerboard] * 6)

            output_prefix = run_dir / "analysis" / "ordering"
            args = build_parser().parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--start-fraction",
                    "0",
                    "--step",
                    "1",
                    "--n-blocks",
                    "2",
                    "--output-prefix",
                    str(output_prefix),
                    "--no-plots",
                ]
            )
            summary, blocks = run(args)

            self.assertEqual([row["temperature_K"] for row in summary], [300.0, 600.0])
            self.assertEqual(len(blocks), 4)
            self.assertAlmostEqual(float(summary[0]["layer_polarization_mean"]), 0.5)
            self.assertAlmostEqual(float(summary[1]["layer_polarization_mean"]), 0.0)
            self.assertEqual(int(summary[0]["intralayer_coordination"]), 2)
            self.assertEqual(int(summary[0]["interlayer_coordination"]), 1)
            self.assertEqual(int(summary[0]["n_alloy_sites"]), 8)
            self.assertAlmostEqual(
                float(summary[0]["alpha_finite_random_baseline"]), -1.0 / 7.0
            )
            self.assertAlmostEqual(
                float(summary[0]["layer_polarization_finite_random_baseline"]),
                9.0 / 35.0,
            )
            self.assertIn("alpha_intralayer_1nn_mean", summary[0])
            self.assertIn("alpha_interlayer_nearest_mean", summary[0])
            self.assertLess(float(summary[1]["alpha_intralayer_1nn_mean"]), 0.0)

            summary_path = Path(f"{output_prefix}_summary.csv")
            self.assertTrue(summary_path.is_file())
            with summary_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2)
            self.assertAlmostEqual(
                float(csv_rows[0]["alpha_finite_random_baseline"]), -1.0 / 7.0
            )
            self.assertAlmostEqual(
                float(csv_rows[0]["layer_polarization_finite_random_baseline"]),
                9.0 / 35.0,
            )


if __name__ == "__main__":
    unittest.main()
