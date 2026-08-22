import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.io import read, write

from gcmc.analysis import LocalAdsorptionMotifAnalyzer
from gcmc.analysis.adsorption_motif_plots import (
    _family_support_fractions,
    _run_fraction_table,
)
from gcmc.analysis.adsorption_motifs_cli import main as motif_cli_main
from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.utils import get_hollow_xy


def _make_reference_slab() -> Atoms:
    a1 = np.array([3.0, 0.0])
    a2 = np.array([1.5, 2.598076211353316])
    top_xy = np.asarray(
        [i * a1 + j * a2 for i in range(2) for j in range(2)],
        dtype=float,
    )
    cell = np.array(
        [[6.0, 0.0, 0.0], [3.0, 5.196152422706632, 0.0], [0.0, 0.0, 20.0]]
    )
    top_z = np.array([8.95, 8.10, 8.85, 8.05])
    top = np.column_stack((top_xy, top_z))
    hollow_xy = get_hollow_xy(top_xy, cell)

    symbols = ["Ti", "Zr", "Ti", "Zr"]
    positions = [position.copy() for position in top]
    for xy in hollow_xy:
        symbols.append("C")
        positions.append(np.array([xy[0], xy[1], 6.5]))
    for local_index, xy in enumerate(hollow_xy):
        symbols.append("Ti" if local_index % 2 == 0 else "Zr")
        positions.append(np.array([xy[0], xy[1], 5.0]))
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )


def _add_oh(
    slab: Atoms,
    descriptor: dict[str, object],
    *,
    anchor_z_shift: float = 0.0,
) -> Atoms:
    atoms = slab.copy()
    anchor = np.array(
        [
            float(descriptor["site_x_A"]),
            float(descriptor["site_y_A"]),
            float(descriptor["suggested_z_A"]) + float(anchor_z_shift),
        ]
    )
    atoms.append(Atom("O", anchor))
    atoms.append(Atom("H", anchor + np.array([0.0, 0.0, 0.98])))
    tags = np.zeros(len(atoms), dtype=int)
    tags[-2:] = ADSORBATE_TAG_OFFSET
    atoms.set_tags(tags)
    return atoms


class TestLocalAdsorptionMotifs(unittest.TestCase):
    def setUp(self):
        self.reference = _make_reference_slab()
        self.analyzer = LocalAdsorptionMotifAnalyzer(
            site_elements=("Ti", "Zr"),
            substrate_elements=("Ti", "Zr", "C"),
            functional_elements=(),
            site_types=("atop", "bridge", "fcc", "hcp"),
            max_site_distance_A=0.75,
            max_anchor_support_distance_A=3.0,
        )

    def _descriptors_by_family(self):
        descriptors = self.analyzer._build_reference_site_descriptors(self.reference)
        selected = {}
        for descriptor in descriptors:
            family, _ = self.analyzer._coordination_family(
                str(descriptor["site_type"]), int(descriptor["support_size"])
            )
            selected.setdefault(family, descriptor)
        return selected

    def test_classifies_coordination_families_and_transitions(self):
        descriptors = self._descriptors_by_family()
        self.assertTrue({"atop", "bridge", "hollow"}.issubset(descriptors))

        atop = _add_oh(self.reference, descriptors["atop"])
        atop_support = int(str(descriptors["atop"]["support_indices"]).split()[0])
        atop.positions[atop_support, 2] += 0.4
        frames = [
            atop,
            _add_oh(self.reference, descriptors["bridge"]),
            _add_oh(self.reference, descriptors["hollow"]),
            _add_oh(self.reference, descriptors["atop"], anchor_z_shift=4.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            traj = Path(tmpdir) / "replica_300K.traj"
            write(traj, frames)
            result = self.analyzer.analyze_trajectory(
                traj,
                reference=self.reference,
            )

        rows = result["local_motif_frames"]
        self.assertEqual(
            [row["adsorption_motif"] for row in rows],
            ["atop", "bridge", "hollow", "detached"],
        )
        self.assertEqual([row["coordination_n"] for row in rows], [1, 2, 3, 0])
        self.assertAlmostEqual(
            float(rows[0]["support_outward_displacement_A_max"]),
            0.4,
            places=6,
        )
        families = {
            str(row["adsorption_motif"]): float(row["population"])
            for row in result["adsorption_family_summary"]
        }
        self.assertEqual(set(families), {"atop", "bridge", "hollow", "detached"})
        self.assertTrue(all(abs(value - 0.25) < 1.0e-12 for value in families.values()))
        transitions = {
            (row["from_motif"], row["to_motif"]): row["count"]
            for row in result["adsorption_motif_transitions"]
        }
        self.assertEqual(transitions[("atop", "bridge")], 1)
        self.assertEqual(transitions[("bridge", "hollow")], 1)
        self.assertEqual(transitions[("hollow", "detached")], 1)

    def test_cli_discovers_trajectory_and_writes_aggregate_outputs(self):
        descriptor = self._descriptors_by_family()["bridge"]
        frame = _add_oh(self.reference, descriptor)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "run" / "results"
            results.mkdir(parents=True)
            write(results / "replica_300K.traj", [frame])
            write(results / "adsorbate_pt_initial.traj", [_add_oh(self.reference, descriptor)])
            out_dir = root / "analysis"

            motif_cli_main(
                [
                    "--traj",
                    str(root / "run"),
                    "--site-elements",
                    "Ti",
                    "Zr",
                    "--substrate-elements",
                    "Ti",
                    "Zr",
                    "C",
                    "--functional-elements",
                    "O",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            family_rows = read(results / "replica_300K.traj", ":")
            self.assertEqual(len(family_rows), 1)
            self.assertTrue((out_dir / "all_families.csv").is_file())
            self.assertTrue((out_dir / "all_patterns.csv").is_file())
            self.assertTrue((out_dir / "all_per_frame.csv").is_file())
            self.assertGreater((out_dir / "motif_distribution.pdf").stat().st_size, 0)
            self.assertGreater((out_dir / "motif_distribution.png").stat().st_size, 0)
            self.assertIn("bridge", (out_dir / "all_families.csv").read_text())

    def test_family_support_color_is_conditional_on_family_population(self):
        rows = [
            {
                "temperature_K": 300.0,
                "traj": "run_a",
                "adsorption_motif": "atop",
                "site_type": "atop",
                "support_key": support,
            }
            for support in ("Ti1", "Ti1", "Ti1", "Zr1")
        ]
        rows.extend(
            [
                {
                    "temperature_K": 300.0,
                    "traj": "run_b",
                    "adsorption_motif": "atop",
                    "site_type": "atop",
                    "support_key": "Zr1",
                },
                {
                    "temperature_K": 300.0,
                    "traj": "run_b",
                    "adsorption_motif": "bridge",
                    "site_type": "bridge",
                    "support_key": "Ti1+Zr1",
                },
            ]
        )
        temperatures, keys, fractions = _run_fraction_table(
            rows,
            key_fields=("adsorption_motif",),
        )
        families = sorted(key[0] for key in keys)
        compositions = _family_support_fractions(
            rows,
            families=families,
            temperatures=temperatures,
            trajectories={300.0: ["run_a", "run_b"]},
            family_fractions=fractions,
            support_color_elements=("Ti", "Zr"),
        )

        self.assertAlmostEqual(compositions["atop"][0], 0.5)
        self.assertAlmostEqual(compositions["bridge"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
