import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from ase import Atoms, units
from ase.build import molecule
from ase.io import read, write
from ase.thermochemistry import IdealGasThermo
from gcmc.constants import ADSORBATE_TAG_OFFSET
from reaction import (
    OERCHESummarizer,
    ReferenceThermoWorkflow,
    ReactionCandidateGenerator,
    ReactionLocalCMCWorkflow,
    ReactionPostProcessingWorkflow,
    ReactionStateRelaxer,
    ReactionStateVibrationWorkflow,
    aggregate_parent_site_rows,
    canonical_parent_site_id,
    load_reaction_postprocess_config,
    load_reference_thermo_config,
    run_reference_thermo_stage,
    site_directory_name,
)


class ReactionWorkflowTests(unittest.TestCase):
    def test_canonical_parent_site_id_sorts_support_indices(self):
        row = {"site_type": "fcc", "support_indices": "385 320 380"}
        self.assertEqual(canonical_parent_site_id(row), "fcc:320-380-385")

    def test_site_directory_name_is_stable_and_path_safe(self):
        self.assertEqual(
            site_directory_name("fcc:320-380-385"),
            "fcc_320-380-385",
        )

    def test_parent_stability_screen_filters_candidate_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "candidate_manifest.csv"
            manifest.write_text(
                "site_id,state_dir,species,candidates_csv,candidates_traj\n"
                "site_a,01_OH,OH,a_oh.csv,a_oh.traj\n"
                "site_a,02_O,O,a_o.csv,a_o.traj\n"
                "site_b,01_OH,OH,b_oh.csv,b_oh.traj\n"
                "site_b,02_O,O,b_o.csv,b_o.traj\n"
            )
            vib = root / "vibration_summary.csv"
            vib.write_text(
                "site_id,state_dir,candidate_id,ready\n"
                "site_a,01_OH,a_oh,True\n"
                "site_b,01_OH,b_oh,False\n"
            )
            cfg = SimpleNamespace(output_dir=str(root))
            workflow = ReactionPostProcessingWorkflow(cfg)

            filtered = workflow._filter_candidate_manifest_by_parent_vibrations(
                manifest,
                vib,
                "01_OH",
                {"output_manifest": "stable.csv"},
            )

            with filtered.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["site_id"] for row in rows}, {"site_a"})
            self.assertEqual({row["state_dir"] for row in rows}, {"01_OH", "02_O"})

    def test_aggregate_parent_site_rows_combines_equivalent_support_orderings(self):
        rows = [
            {
                "site_type": "fcc",
                "support_indices": "385 320 380",
                "support_key": "Zr2+Mo1",
                "shell1_key": "A",
                "shell2_key": "B",
                "functional_count": "2",
                "motif_key": "m1",
                "temperature_K": 303.0,
                "_temperature_label": "303K",
                "anchor_site_xy_dist_A": 0.8,
                "anchor_support_min_dist_A": 2.1,
                "anchor_z_offset_A": 1.0,
            },
            {
                "site_type": "fcc",
                "support_indices": "320 385 380",
                "support_key": "Zr2+Mo1",
                "shell1_key": "A",
                "shell2_key": "B",
                "functional_count": "2",
                "motif_key": "m1",
                "temperature_K": 312.0,
                "_temperature_label": "312K",
                "anchor_site_xy_dist_A": 1.0,
                "anchor_support_min_dist_A": 2.0,
                "anchor_z_offset_A": 1.2,
            },
            {
                "site_type": "hcp",
                "support_indices": "1 2 3",
                "support_key": "Ti3",
                "shell1_key": "C",
                "shell2_key": "D",
                "functional_count": "1",
                "motif_key": "m2",
                "temperature_K": 303.0,
                "_temperature_label": "303K",
                "anchor_site_xy_dist_A": 0.7,
                "anchor_support_min_dist_A": 2.2,
                "anchor_z_offset_A": 0.9,
            },
        ]

        summary = aggregate_parent_site_rows(
            rows,
            temperature_labels=("303K", "312K"),
        )

        self.assertEqual(summary[0]["site_id"], "fcc:320-380-385")
        self.assertEqual(summary[0]["samples_total"], 2)
        self.assertAlmostEqual(summary[0]["population_total"], 2 / 3)
        self.assertAlmostEqual(summary[0]["population_303K"], 0.5)
        self.assertAlmostEqual(summary[0]["population_312K"], 1.0)

    def test_load_reaction_postprocess_config_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg_path = root / "post.yaml"
            cfg_path.write_text(
                """
parent:
  traj: [replica_303K.traj]
site_analysis:
  site_elements: [Ti, Zr, Mo]
  substrate_elements: [Ti, Zr, Mo, C]
  functional_elements: [O]
  site_types: [atop, fcc, hcp]
output:
  output_dir: out
  out_prefix: oh_parent
state_relaxation:
  log_file: relax.log
che:
  vibration_summary_csv: vib.csv
""".strip()
            )

            cfg = load_reaction_postprocess_config(cfg_path)

        self.assertEqual(cfg.parent_traj, [str(root / "replica_303K.traj")])
        self.assertEqual(cfg.output_dir, str(root / "out"))
        self.assertEqual(cfg.out_prefix, "oh_parent")
        self.assertEqual(cfg.site_elements, ("Ti", "Zr", "Mo"))
        self.assertEqual(cfg.reaction_state_dirs, ("00_clean", "01_OH", "02_O", "03_OOH"))
        self.assertTrue(cfg.candidate_generation["enabled"])
        self.assertFalse(cfg.state_relaxation["enabled"])
        self.assertFalse(cfg.vibrations["enabled"])
        self.assertEqual(cfg.state_relaxation["log_file"], "relax.log")
        self.assertFalse(cfg.reference_thermo["enabled"])
        self.assertEqual(cfg.che["vibration_summary_csv"], str(root / "vib.csv"))

    def test_load_reference_thermo_config_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg_path = root / "ref.yaml"
            cfg_path.write_text(
                """
reference:
  molecule: H2
calculator:
  model_file: model.json
output:
  output_dir: ref_out
relaxation:
  log_file: relax.log
  progress_log: ref.log
""".strip()
            )

            cfg = load_reference_thermo_config(cfg_path)

        self.assertEqual(cfg.molecule, "H2")
        self.assertEqual(cfg.model_file, str(root / "model.json"))
        self.assertEqual(cfg.output_dir, str(root / "ref_out"))
        self.assertEqual(cfg.relaxation["log_file"], "relax.log")
        self.assertEqual(cfg.relaxation["progress_log"], "ref.log")

    def test_candidate_generator_writes_clean_parent_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            reps_traj = site_dir / "representatives.traj"
            reps_csv = site_dir / "representatives.csv"
            atoms = Atoms(
                "PtOH",
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 0.0, 2.8]],
            )
            atoms.set_tags([0, 7, 7])
            write(str(reps_traj), [atoms])
            reps_csv.write_text(
                "representative_traj,representative_frame,"
                "representative_anchor_index,representative_rank_within_group\n"
                f"{reps_traj},0,1,0\n"
            )
            site_manifest = root / "site_manifest.csv"
            site_manifest.write_text(
                "site_id,site_dir,reactions_dir,site_representatives_traj,"
                "site_representatives_csv,reaction_state_dirs\n"
                f"fcc:1-2-3,{site_dir},{reactions_dir},{reps_traj},{reps_csv},00_clean\n"
            )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                reaction_state_dirs=("00_clean",),
                candidate_generation={"enabled": True},
            )

            outputs = ReactionCandidateGenerator(cfg).generate(site_manifest)

            self.assertEqual(
                outputs["candidate_manifest_csv"],
                str(root / "candidate_manifest.csv"),
            )
            clean_traj = reactions_dir / "00_clean" / "candidates.traj"
            clean = read(str(clean_traj), index=":")[0]
            self.assertEqual(clean.get_chemical_formula(), "Pt")
            with (reactions_dir / "00_clean" / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["atom_origin_indices"], "0")
            self.assertEqual(rows[0]["atom_is_adsorbate"], "0")

    def test_candidate_generator_can_place_o_on_parent_stripped_slab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            reps_traj = site_dir / "representatives.traj"
            reps_csv = site_dir / "representatives.csv"
            atoms = Atoms(
                "Pt4OH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [2.8, 0.0, 0.0],
                    [0.0, 2.8, 0.0],
                    [2.8, 2.8, 0.0],
                    [0.0, 0.0, 2.8],
                    [0.0, 0.0, 3.8],
                ],
                cell=[5.6, 5.6, 10.0],
                pbc=[True, True, False],
            )
            atoms.set_tags([0, 0, 0, 0, 7, 7])
            write(str(reps_traj), [atoms])
            reps_csv.write_text(
                "representative_traj,representative_frame,"
                "representative_anchor_index,representative_rank_within_group\n"
                f"{reps_traj},0,4,0\n"
            )
            site_manifest = root / "site_manifest.csv"
            site_manifest.write_text(
                "site_id,site_dir,reactions_dir,site_representatives_traj,"
                "site_representatives_csv,reaction_state_dirs,site_type,"
                "support_indices_sorted\n"
                f"atop:0,{site_dir},{reactions_dir},{reps_traj},{reps_csv},"
                "02_O,atop,0\n"
            )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                reaction_state_dirs=("02_O",),
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                candidate_generation={
                    "enabled": True,
                    "child_slab_mode": "parent_stripped",
                    "include_original_o_site": True,
                    "include_nearby_o_sites": False,
                },
            )

            ReactionCandidateGenerator(cfg).generate(site_manifest)

            o_traj = reactions_dir / "02_O" / "candidates.traj"
            candidate = read(str(o_traj), index=":")[0]
            self.assertEqual(candidate.get_chemical_formula(), "OPt4")
            self.assertAlmostEqual(candidate.positions[4, 2], 1.8)
            with (reactions_dir / "02_O" / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["candidate_kind"], "direct_parent_stripped")
            self.assertEqual(rows[0]["child_slab_mode"], "parent_stripped")

    def test_oer_che_summarizer_writes_site_free_energies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            (site_dir / "representatives.csv").write_text(
                "site_id,population_total\nfcc:1-2-3,0.25\n"
            )
            manifest = root / "oh_parent_sites_candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, energy in (
                    ("00_clean", "CLEAN", 0.0),
                    ("01_OH", "OH", 2.0),
                    ("02_O", "O", 3.0),
                    ("03_OOH", "OOH", 4.0),
                ):
                    state_dir = reactions_dir / state
                    state_dir.mkdir(parents=True)
                    (state_dir / "candidates.csv").write_text("candidate_id\ntest\n")
                    (state_dir / "energies.csv").write_text(
                        "candidate_id,candidate_kind,converged,energy_eV,"
                        "fmax_eV_A,nsteps,error\n"
                        f"{state}_best,test,True,{energy},0.01,5,\n"
                    )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(state_dir / "candidates.traj"),
                            "candidates_csv": str(state_dir / "candidates.csv"),
                        }
                    )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            self.assertTrue(Path(outputs["oer_routes_csv"]).exists())
            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            for column in (
                "route_id",
                "route_mode",
                "o_basin_id",
                "ooh_basin_id",
                "route_weight",
                "DeltaG1_eV",
                "overpotential_V",
            ):
                self.assertIn(column, rows[0])
            self.assertEqual(rows[0]["ready_min"], "True")
            self.assertAlmostEqual(float(rows[0]["DeltaG1_min_eV"]), 2.0)
            self.assertAlmostEqual(float(rows[0]["overpotential_min_V"]), 0.77)

    def test_oer_che_summarizer_can_use_lowest_shared_clean_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "oh_parent_sites_candidate_manifest.csv"
            fieldnames = [
                "site_id",
                "site_dir",
                "state_dir",
                "species",
                "n_candidates",
                "candidates_traj",
                "candidates_csv",
            ]
            site_energies = {
                "fcc:1-2-3": {
                    "00_clean": 5.0,
                    "01_OH": 8.0,
                    "02_O": 9.0,
                    "03_OOH": 10.0,
                },
                "fcc:4-5-6": {
                    "00_clean": 1.0,
                    "01_OH": 3.0,
                    "02_O": 4.0,
                    "03_OOH": 5.0,
                },
            }
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for site_id, energies in site_energies.items():
                    site_dir = root / "sites" / site_directory_name(site_id)
                    reactions_dir = site_dir / "reactions"
                    site_dir.mkdir(parents=True)
                    for state, energy in energies.items():
                        state_dir = reactions_dir / state
                        state_dir.mkdir(parents=True)
                        (state_dir / "candidates.csv").write_text("candidate_id\ntest\n")
                        (state_dir / "energies.csv").write_text(
                            "candidate_id,candidate_kind,converged,energy_eV,"
                            "fmax_eV_A,nsteps,error\n"
                            f"{state}_best,test,True,{energy},0.01,5,\n"
                        )
                        writer.writerow(
                            {
                                "site_id": site_id,
                                "site_dir": str(site_dir),
                                "state_dir": state,
                                "species": state.removeprefix("00_").removeprefix("01_"),
                                "n_candidates": 1,
                                "candidates_traj": str(state_dir / "candidates.traj"),
                                "candidates_csv": str(state_dir / "candidates.csv"),
                            }
                        )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "clean_reference_mode": "lowest_energy",
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = {row["site_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(
                rows["fcc:1-2-3"]["clean_reference_site_id"],
                "fcc:4-5-6",
            )
            self.assertAlmostEqual(float(rows["fcc:1-2-3"]["DeltaG1_min_eV"]), 7.0)

    def test_oer_che_summarizer_uses_harmonic_free_energies_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, energy in (
                    ("00_clean", "CLEAN", 0.0),
                    ("01_OH", "OH", 2.0),
                    ("02_O", "O", 3.0),
                    ("03_OOH", "OOH", 4.0),
                ):
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    (state_path / "candidates.csv").write_text(
                        "candidate_id,candidate_kind\n"
                        f"{state}_best,test\n"
                    )
                    (state_path / "energies.csv").write_text(
                        "candidate_id,candidate_kind,converged,energy_eV,"
                        "fmax_eV_A,nsteps,error\n"
                        f"{state}_best,test,True,{energy},0.01,5,\n"
                    )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            (root / "vibration_summary.csv").write_text(
                "site_id,state_dir,candidate_id,ready,harmonic_free_energy_eV,harmonic_correction_eV\n"
                "fcc:1-2-3,00_clean,00_clean_best,True,0.10,0.10\n"
                "fcc:1-2-3,01_OH,01_OH_best,True,2.20,0.20\n"
                "fcc:1-2-3,02_O,02_O_best,True,3.30,0.30\n"
                "fcc:1-2-3,03_OOH,03_OOH_best,True,4.40,0.40\n"
            )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "use_vibrational_free_energies": True,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["DeltaG1_min_eV"]), 2.10)
            self.assertAlmostEqual(float(rows[0]["DeltaG1_min_electronic_eV"]), 2.0)
            with Path(outputs["oer_states_csv"]).open() as handle:
                best_rows = {
                    (row["state_label"], row["min_candidate_id"]): row
                    for row in csv.DictReader(handle)
                }
            self.assertEqual(best_rows[("oh", "01_OH_best")]["min_energy_source"], "harmonic")
            self.assertAlmostEqual(float(best_rows[("oh", "01_OH_best")]["electronic_energy_eV"]), 2.0)
            self.assertAlmostEqual(float(best_rows[("oh", "01_OH_best")]["harmonic_correction_eV"]), 0.20)
            self.assertAlmostEqual(
                float(best_rows[("oh", "01_OH_best")]["min_electronic_free_energy_eV"]),
                2.0,
            )

    def test_oer_che_closure_uses_harmonic_free_energy_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, electronic, harmonic in (
                    ("00_clean", "CLEAN", 0.0, 0.10),
                    ("01_OH", "OH", 2.0, 2.20),
                    ("02_O", "O", 3.0, 3.30),
                    ("03_OOH", "OOH", 4.0, 4.40),
                ):
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    (state_path / "candidates.csv").write_text("candidate_id\ntest\n")
                    (state_path / "energies.csv").write_text(
                        "candidate_id,candidate_kind,converged,energy_eV,"
                        "fmax_eV_A,nsteps,error\n"
                        f"{state}_best,test,True,{electronic},0.01,5,\n"
                    )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )
            (root / "vibration_summary.csv").write_text(
                "site_id,state_dir,candidate_id,ready,harmonic_free_energy_eV,harmonic_correction_eV\n"
                "fcc:1-2-3,00_clean,00_clean_best,True,0.10,0.10\n"
                "fcc:1-2-3,01_OH,01_OH_best,True,2.20,0.20\n"
                "fcc:1-2-3,02_O,02_O_best,True,3.30,0.30\n"
                "fcc:1-2-3,03_OOH,03_OOH_best,True,4.40,0.40\n"
            )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "use_vibrational_free_energies": True,
                    "oer_reference_mode": "closure",
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                row = next(csv.DictReader(handle))
            deltas = [
                float(row[f"DeltaG{idx}_min_eV"])
                for idx in range(1, 5)
            ]
            self.assertAlmostEqual(deltas[0], 2.10)
            self.assertAlmostEqual(sum(deltas), 4.92)
            self.assertAlmostEqual(deltas[3], 4.92 - sum(deltas[:3]))

    def test_oer_che_summarizer_excludes_failed_vibrations_when_harmonic_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            fieldnames = [
                "site_id",
                "site_dir",
                "state_dir",
                "species",
                "n_candidates",
                "candidates_traj",
                "candidates_csv",
            ]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for state, species, rows in (
                    ("00_clean", "CLEAN", [("clean", 0.0)]),
                    ("01_OH", "OH", [("oh", 2.0)]),
                    ("02_O", "O", [("o", 3.0)]),
                    (
                        "03_OOH",
                        "OOH",
                        [("ooh_good", 4.0), ("ooh_failed_vib", 1.0)],
                    ),
                ):
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    (state_path / "candidates.csv").write_text("candidate_id\n")
                    with (state_path / "energies.csv").open("w", newline="") as ehandle:
                        writer_e = csv.DictWriter(
                            ehandle,
                            fieldnames=[
                                "candidate_id",
                                "candidate_kind",
                                "converged",
                                "energy_eV",
                                "fmax_eV_A",
                                "nsteps",
                                "error",
                            ],
                        )
                        writer_e.writeheader()
                        for candidate_id, energy in rows:
                            writer_e.writerow(
                                {
                                    "candidate_id": candidate_id,
                                    "candidate_kind": "test",
                                    "converged": True,
                                    "energy_eV": energy,
                                    "fmax_eV_A": 0.01,
                                    "nsteps": 5,
                                    "error": "",
                                }
                            )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": len(rows),
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            (root / "vibration_summary.csv").write_text(
                "site_id,state_dir,candidate_id,ready,harmonic_free_energy_eV,harmonic_correction_eV\n"
                "fcc:1-2-3,00_clean,clean,True,0.10,0.10\n"
                "fcc:1-2-3,01_OH,oh,True,2.20,0.20\n"
                "fcc:1-2-3,02_O,o,True,3.30,0.30\n"
                "fcc:1-2-3,03_OOH,ooh_good,True,4.40,0.40\n"
                "fcc:1-2-3,03_OOH,ooh_failed_vib,False,,\n"
            )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "use_vibrational_free_energies": True,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_states_csv"]).open() as handle:
                rows = {
                    row["state_label"]: row
                    for row in csv.DictReader(handle)
                }
            self.assertEqual(rows["ooh"]["min_candidate_id"], "ooh_good")
            self.assertEqual(rows["ooh"]["min_energy_source"], "harmonic")
            self.assertAlmostEqual(float(rows["ooh"]["min_free_energy_eV"]), 4.40)

    def test_oer_che_summarizer_can_use_ideal_gas_thermo_for_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, energy in (
                    ("00_clean", "CLEAN", 0.0),
                    ("01_OH", "OH", 2.0),
                    ("02_O", "O", 3.0),
                    ("03_OOH", "OOH", 4.0),
                ):
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    (state_path / "candidates.csv").write_text(
                        "candidate_id,candidate_kind\n"
                        f"{state}_best,test\n"
                    )
                    (state_path / "energies.csv").write_text(
                        "candidate_id,candidate_kind,converged,energy_eV,"
                        "fmax_eV_A,nsteps,error\n"
                        f"{state}_best,test,True,{energy},0.01,5,\n"
                    )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            h2_energy = -6.0
            h2o_energy = -14.0
            thermo_h2 = IdealGasThermo(
                vib_energies=[0.54],
                geometry="linear",
                potentialenergy=h2_energy,
                atoms=molecule("H2"),
                symmetrynumber=2,
                spin=0.0,
            )
            thermo_h2o = IdealGasThermo(
                vib_energies=[0.20, 0.45, 0.47],
                geometry="nonlinear",
                potentialenergy=h2o_energy,
                atoms=molecule("H2O"),
                symmetrynumber=2,
                spin=0.0,
            )
            h2_g = thermo_h2.get_gibbs_energy(300.0, 101325.0, verbose=False)
            h2o_g = thermo_h2o.get_gibbs_energy(300.0, 101325.0, verbose=False)
            expected_dg1 = 2.0 - 0.0 - h2o_g + 0.5 * h2_g

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "h2_energy_eV": h2_energy,
                    "h2o_energy_eV": h2o_energy,
                    "h2_thermo": {
                        "enabled": True,
                        "temperature_K": 300.0,
                        "pressure_Pa": 101325.0,
                        "geometry": "linear",
                        "symmetrynumber": 2,
                        "spin": 0.0,
                        "vib_energies_eV": [0.54],
                    },
                    "h2o_thermo": {
                        "enabled": True,
                        "temperature_K": 300.0,
                        "pressure_Pa": 101325.0,
                        "geometry": "nonlinear",
                        "symmetrynumber": 2,
                        "spin": 0.0,
                        "vib_energies_eV": [0.20, 0.45, 0.47],
                    },
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["DeltaG1_min_eV"]), expected_dg1)

    def test_oer_che_summarizer_can_use_explicit_o2_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, energy in (
                    ("00_clean", "CLEAN", 0.0),
                    ("01_OH", "OH", 2.0),
                    ("02_O", "O", 3.0),
                    ("03_OOH", "OOH", 4.0),
                ):
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    (state_path / "candidates.csv").write_text("candidate_id\ntest\n")
                    (state_path / "energies.csv").write_text(
                        "candidate_id,candidate_kind,converged,energy_eV,"
                        "fmax_eV_A,nsteps,error\n"
                        f"{state}_best,test,True,{energy},0.01,5,\n"
                    )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "oer_reference_mode": "explicit_o2",
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "o2_energy_eV": 10.0,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["DeltaG4_min_eV"]), 6.0)
            self.assertEqual(rows[0]["limiting_step_min"], "4")
            self.assertAlmostEqual(float(rows[0]["overpotential_min_V"]), 4.77)

    def test_oer_che_summarizer_writes_explicit_route_ensemble(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "fcc_1-2-3"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                state_rows = {
                    "00_clean": [0.0],
                    "01_OH": [2.0],
                    "02_O": [3.0, 3.0],
                    "03_OOH": [4.0],
                }
                for state, energies in state_rows.items():
                    state_dir = reactions_dir / state
                    state_dir.mkdir(parents=True)
                    (state_dir / "candidates.csv").write_text("candidate_id\ntest\n")
                    with (state_dir / "energies.csv").open("w", newline="") as energy_handle:
                        energy_writer = csv.DictWriter(
                            energy_handle,
                            fieldnames=[
                                "candidate_id",
                                "candidate_kind",
                                "converged",
                                "energy_eV",
                                "fmax_eV_A",
                                "nsteps",
                                "error",
                            ],
                        )
                        energy_writer.writeheader()
                        for index, energy in enumerate(energies):
                            energy_writer.writerow(
                                {
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "converged": "True",
                                    "energy_eV": energy,
                                    "fmax_eV_A": 0.01,
                                    "nsteps": 5,
                                    "error": "",
                                }
                            )
                    writer.writerow(
                        {
                            "site_id": "fcc:1-2-3",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": state.split("_", 1)[-1],
                            "n_candidates": len(energies),
                            "candidates_traj": str(state_dir / "candidates.traj"),
                            "candidates_csv": str(state_dir / "candidates.csv"),
                        }
                    )
            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                che={
                    "enabled": True,
                    "route_temperature_K": 300.0,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            self.assertTrue(Path(outputs["oer_routes_csv"]).exists())
            self.assertTrue(Path(outputs["oer_ensemble_csv"]).exists())
            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["n_O_candidates"], "2")
            self.assertAlmostEqual(
                sum(float(row["route_weight"]) for row in rows),
                1.0,
            )
            self.assertAlmostEqual(float(rows[0]["DeltaG2_eV"]), 1.0)
            with Path(outputs["oer_ensemble_csv"]).open() as handle:
                ensemble = list(csv.DictReader(handle))
            self.assertEqual(
                list(ensemble[0].keys()),
                [
                    "source",
                    "n_ready_routes",
                    "population_sum_ready",
                    "population_missing",
                    "route_weight_model",
                    "basin_cluster_mode",
                    "basin_energy_cluster_tol_eV",
                    "route_weighted_mean_overpotential_V",
                    "min_site_overpotential_V",
                    "min_overpotential_route_id",
                    "min_overpotential_site_id",
                    "dominant_route_id",
                    "dominant_route_site_id",
                    "dominant_route_weight",
                ],
            )
            self.assertAlmostEqual(
                float(ensemble[0]["route_weighted_mean_overpotential_V"]),
                0.77,
            )
            self.assertEqual(
                ensemble[0]["route_weight_model"],
                "empirical_conditional_route_probability",
            )

    def test_oer_che_geometry_clustering_collapses_duplicate_basin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                state_rows = {
                    "00_clean": ("CLEAN", [0.0]),
                    "01_OH": ("OH", [2.0]),
                    "02_O": ("O", [3.0, 3.0]),
                    "03_OOH": ("OOH", [4.0]),
                }
                for state, (species, energies) in state_rows.items():
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    with (state_path / "candidates.csv").open("w", newline="") as ch:
                        candidate_writer = csv.DictWriter(
                            ch,
                            fieldnames=[
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                            ],
                        )
                        candidate_writer.writeheader()
                        for index in range(len(energies)):
                            candidate_writer.writerow(
                                {
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                }
                            )
                    with (state_path / "energies.csv").open("w", newline="") as eh:
                        energy_writer = csv.DictWriter(
                            eh,
                            fieldnames=[
                                "candidate_index",
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                                "converged",
                                "energy_eV",
                                "fmax_eV_A",
                                "nsteps",
                                "adsorbate_intact",
                                "error",
                            ],
                        )
                        energy_writer.writeheader()
                        for index, energy in enumerate(energies):
                            energy_writer.writerow(
                                {
                                    "candidate_index": index,
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                    "converged": True,
                                    "energy_eV": energy,
                                    "fmax_eV_A": 0.01,
                                    "nsteps": 5,
                                    "adsorbate_intact": True,
                                    "error": "",
                                }
                            )
                    writer.writerow(
                        {
                            "site_id": "atop:0",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": len(energies),
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            slab = [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [0.0, 2.8, 0.0], [2.8, 2.8, 0.0]]
            duplicate_o = []
            for _ in range(2):
                atoms = Atoms(
                    "Pt4O",
                    positions=[*slab, [0.0, 0.0, 1.8]],
                    cell=[5.6, 5.6, 10.0],
                    pbc=[True, True, False],
                )
                atoms.set_tags([0, 0, 0, 0, 1_000_000])
                duplicate_o.append(atoms)
            write(str(reactions_dir / "02_O" / "relaxed.traj"), duplicate_o)

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                che={
                    "enabled": True,
                    "basin_cluster_mode": "geometry",
                    "basin_energy_cluster_tol_eV": 0.0,
                    "route_temperature_K": 300.0,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["n_O_candidates"], "1")
            self.assertAlmostEqual(float(row["DeltaG2_eV"]), 1.0)

    def test_oer_che_geometry_clustering_keeps_distinct_local_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                state_rows = {
                    "00_clean": ("CLEAN", [0.0]),
                    "01_OH": ("OH", [2.0]),
                    "02_O": ("O", [3.0, 3.1]),
                    "03_OOH": ("OOH", [4.0]),
                }
                for state, (species, energies) in state_rows.items():
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    with (state_path / "candidates.csv").open("w", newline="") as ch:
                        candidate_writer = csv.DictWriter(
                            ch,
                            fieldnames=[
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                            ],
                        )
                        candidate_writer.writeheader()
                        for index in range(len(energies)):
                            candidate_writer.writerow(
                                {
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                }
                            )
                    with (state_path / "energies.csv").open("w", newline="") as eh:
                        energy_writer = csv.DictWriter(
                            eh,
                            fieldnames=[
                                "candidate_index",
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                                "converged",
                                "energy_eV",
                                "fmax_eV_A",
                                "nsteps",
                                "adsorbate_intact",
                                "error",
                            ],
                        )
                        energy_writer.writeheader()
                        for index, energy in enumerate(energies):
                            energy_writer.writerow(
                                {
                                    "candidate_index": index,
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                    "converged": True,
                                    "energy_eV": energy,
                                    "fmax_eV_A": 0.01,
                                    "nsteps": 5,
                                    "adsorbate_intact": True,
                                    "error": "",
                                }
                            )
                    writer.writerow(
                        {
                            "site_id": "atop:0",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": len(energies),
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            slab = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [2.8, 2.8, 0.0],
            ]
            distinct_o = []
            for z_offset in (0.0, 0.5):
                atoms = Atoms(
                    "Pt4O",
                    positions=[
                        slab[0],
                        [2.8, 0.0, z_offset],
                        slab[2],
                        slab[3],
                        [0.0, 0.0, 1.8],
                    ],
                    cell=[5.6, 5.6, 10.0],
                    pbc=[True, True, False],
                )
                atoms.set_tags([0, 0, 0, 0, 1_000_000])
                distinct_o.append(atoms)
            write(str(reactions_dir / "02_O" / "relaxed.traj"), distinct_o)

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                che={
                    "enabled": True,
                    "basin_cluster_mode": "geometry",
                    "basin_geometry_rmsd_tol_A": 0.25,
                    "basin_local_env_enabled": True,
                    "basin_local_env_cutoff_A": 3.5,
                    "basin_local_env_rmsd_tol_A": 0.20,
                    "route_temperature_K": 300.0,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["n_O_candidates"], "2")
            self.assertEqual(rows[0]["o_basin_count"], "1")

    def test_oer_che_assigns_basin_weights_from_pt_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            reactions_dir = site_dir / "reactions"
            site_dir.mkdir(parents=True)
            manifest = root / "candidate_manifest.csv"
            slab = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [2.8, 2.8, 0.0],
            ]

            def adsorbate_at(xy):
                atoms = Atoms(
                    "Pt4O",
                    positions=[*slab, [float(xy[0]), float(xy[1]), 1.8]],
                    cell=[5.6, 5.6, 10.0],
                    pbc=[True, True, False],
                )
                atoms.set_tags([0, 0, 0, 0, ADSORBATE_TAG_OFFSET])
                return atoms

            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                state_rows = {
                    "00_clean": ("CLEAN", [0.0]),
                    "01_OH": ("OH", [2.0]),
                    "02_O": ("O", [3.0, 3.1]),
                    "03_OOH": ("OOH", [4.0]),
                }
                for state, (species, energies) in state_rows.items():
                    state_path = reactions_dir / state
                    state_path.mkdir(parents=True)
                    pt_dir = state_path / "local_cmc" / "seed000_pt"
                    pt_dir.mkdir(parents=True)
                    with (state_path / "candidates.csv").open("w", newline="") as ch:
                        candidate_writer = csv.DictWriter(
                            ch,
                            fieldnames=[
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                                "local_cmc_pt_dir",
                                "local_cmc_pt_temperature_K",
                            ],
                        )
                        candidate_writer.writeheader()
                        for index in range(len(energies)):
                            candidate_writer.writerow(
                                {
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                    "local_cmc_pt_dir": str(pt_dir),
                                    "local_cmc_pt_temperature_K": 300.0,
                                }
                            )
                    with (state_path / "energies.csv").open("w", newline="") as eh:
                        energy_writer = csv.DictWriter(
                            eh,
                            fieldnames=[
                                "candidate_index",
                                "candidate_id",
                                "candidate_kind",
                                "anchor_index",
                                "converged",
                                "energy_eV",
                                "fmax_eV_A",
                                "nsteps",
                                "adsorbate_intact",
                                "error",
                            ],
                        )
                        energy_writer.writeheader()
                        for index, energy in enumerate(energies):
                            energy_writer.writerow(
                                {
                                    "candidate_index": index,
                                    "candidate_id": f"{state}_{index}",
                                    "candidate_kind": "test",
                                    "anchor_index": 4,
                                    "converged": True,
                                    "energy_eV": energy,
                                    "fmax_eV_A": 0.01,
                                    "nsteps": 5,
                                    "adsorbate_intact": True,
                                    "error": "",
                                }
                            )
                    writer.writerow(
                        {
                            "site_id": "atop:0",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": len(energies),
                            "candidates_traj": str(state_path / "candidates.traj"),
                            "candidates_csv": str(state_path / "candidates.csv"),
                        }
                    )

            selected_o = [adsorbate_at((0.0, 0.0)), adsorbate_at((2.8, 0.0))]
            write(str(reactions_dir / "02_O" / "candidates.traj"), selected_o)
            write(str(reactions_dir / "02_O" / "relaxed.traj"), selected_o)
            trajectory_frames = [
                adsorbate_at((0.0, 0.0)),
                adsorbate_at((0.05, 0.0)),
                adsorbate_at((0.0, 0.05)),
                adsorbate_at((0.05, 0.05)),
                adsorbate_at((2.8, 0.0)),
            ]
            write(
                str(
                    reactions_dir
                    / "02_O"
                    / "local_cmc"
                    / "seed000_pt"
                    / "replica_300K.traj"
                ),
                trajectory_frames,
            )

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                che={
                    "enabled": True,
                    "basin_cluster_mode": "geometry",
                    "basin_geometry_rmsd_tol_A": 0.25,
                    "basin_local_env_enabled": False,
                    "basin_weight_source": "trajectory",
                    "route_pairing_mode": "cartesian",
                    "route_temperature_K": 300.0,
                    "h2_energy_eV": 0.0,
                    "h2o_energy_eV": 0.0,
                    "total_oer_free_energy_eV": 4.92,
                    "equilibrium_potential_V": 1.23,
                },
            )

            registry_calls = 0

            def counting_registry(*args, **kwargs):
                nonlocal registry_calls
                registry_calls += 1
                from gcmc.utils import build_surface_site_registry

                return build_surface_site_registry(*args, **kwargs)

            with patch(
                "reaction.che.build_surface_site_registry",
                counting_registry,
            ):
                outputs = OERCHESummarizer(cfg).summarize(manifest)

            with Path(outputs["oer_routes_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(registry_calls, 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["o_basin_count"] for row in rows], ["4", "1"])
            self.assertEqual(rows[0]["o_basin_weight_source"], "trajectory")
            self.assertEqual(rows[0]["o_basin_selected_count"], "1")
            self.assertEqual(rows[0]["o_basin_trajectory_total_count"], "5")
            self.assertEqual(rows[0]["o_basin_trajectory_assigned_count"], "5")
            self.assertAlmostEqual(float(rows[0]["o_basin_probability"]), 0.8)
            self.assertAlmostEqual(float(rows[1]["o_basin_probability"]), 0.2)

    def test_reaction_local_cmc_replaces_selected_state_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            reactions_dir = site_dir / "reactions"
            oh_dir = reactions_dir / "01_OH"
            o_dir = reactions_dir / "02_O"
            oh_dir.mkdir(parents=True)
            o_dir.mkdir(parents=True)

            slab_positions = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [2.8, 2.8, 0.0],
            ]
            oh_atoms = Atoms(
                "Pt4OH",
                positions=[*slab_positions, [0.0, 0.0, 1.8], [0.0, 0.0, 2.8]],
                cell=[5.6, 5.6, 10.0],
                pbc=[True, True, False],
            )
            oh_atoms.set_tags([0, 0, 0, 0, 1_000_000, 1_000_000])
            o_atoms = Atoms(
                "Pt4O",
                positions=[*slab_positions, [0.0, 0.0, 1.8]],
                cell=[5.6, 5.6, 10.0],
                pbc=[True, True, False],
            )
            o_atoms.set_tags([0, 0, 0, 0, 1_000_000])
            write(str(oh_dir / "candidates.traj"), [oh_atoms])
            write(str(o_dir / "candidates.traj"), [o_atoms])
            for path, state, species in (
                (oh_dir, "01_OH", "OH"),
                (o_dir, "02_O", "O"),
            ):
                (path / "candidates.csv").write_text(
                    "site_id,state,species,candidate_id,candidate_kind,anchor_index,"
                    "source_representative_rank\n"
                    f"atop:0,{state},{species},{state}_seed,seed,4,0\n"
                )

            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                for state, species, path in (
                    ("01_OH", "OH", oh_dir),
                    ("02_O", "O", o_dir),
                ):
                    writer.writerow(
                        {
                            "site_id": "atop:0",
                            "site_dir": str(site_dir),
                            "state_dir": state,
                            "species": species,
                            "n_candidates": 1,
                            "candidates_traj": str(path / "candidates.traj"),
                            "candidates_csv": str(path / "candidates.csv"),
                        }
                    )

            cfg = SimpleNamespace(
                output_dir=str(root),
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                local_cmc={
                    "enabled": True,
                    "states": ["02_O"],
                    "n_cycles": 1,
                    "sample_interval": 1,
                    "equilibration_cycles": 0,
                    "radius_A": 3.0,
                    "move_mode": "site_hop",
                    "write_debug_trajs": True,
                    "progress_stdout": False,
                    "replace_candidates": True,
                },
            )

            outputs = ReactionLocalCMCWorkflow(cfg).run(manifest)

            self.assertTrue(Path(outputs["local_cmc_manifest_csv"]).exists())
            with (o_dir / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["candidate_kind"], "local_cmc_sample")
            self.assertIn("_localcmc000", rows[0]["candidate_id"])
            with manifest.open() as handle:
                manifest_rows = {row["state_dir"]: row for row in csv.DictReader(handle)}
            self.assertEqual(manifest_rows["02_O"]["n_candidates"], "1")
            local_dir = o_dir / "local_cmc"
            self.assertTrue((local_dir / "seed000_attempted.traj").exists())
            self.assertTrue((local_dir / "seed000_accepted.traj").exists())
            self.assertTrue((local_dir / "seed000_rejected.traj").exists())

    def test_reaction_local_cmc_sequential_rule_regenerates_target_from_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            o_dir = site_dir / "reactions" / "02_O"
            ooh_dir = site_dir / "reactions" / "03_OOH"
            o_dir.mkdir(parents=True)
            ooh_dir.mkdir(parents=True)

            slab_positions = [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 2.8, 0.0],
                [2.8, 2.8, 0.0],
            ]
            o_a = Atoms(
                "Pt4O",
                positions=[*slab_positions, [0.0, 0.0, 1.8]],
                cell=[5.6, 5.6, 10.0],
                pbc=[True, True, False],
            )
            o_b = o_a.copy()
            o_b.positions[4, :2] = [2.8, 0.0]
            for atoms in (o_a, o_b):
                atoms.set_tags([0, 0, 0, 0, ADSORBATE_TAG_OFFSET])
            write(str(o_dir / "candidates.traj"), [o_a, o_b])
            (o_dir / "candidates.csv").write_text(
                "site_id,state,species,candidate_id,candidate_kind,anchor_index\n"
                "atop:0,02_O,O,o_basin_a,local_cmc_pt_sample,4\n"
                "atop:0,02_O,O,o_basin_b,local_cmc_pt_sample,4\n"
            )
            write(str(ooh_dir / "candidates.traj"), [o_a])
            (ooh_dir / "candidates.csv").write_text(
                "site_id,state,species,candidate_id,candidate_kind,anchor_index,"
                "parent_o_candidate_id\n"
                "atop:0,03_OOH,OOH,old_ooh,seed,4,old_o_seed\n"
            )

            o_row = {
                "site_id": "atop:0",
                "state_dir": "02_O",
                "species": "O",
                "candidates_traj": str(o_dir / "candidates.traj"),
                "candidates_csv": str(o_dir / "candidates.csv"),
            }
            ooh_row: dict[str, object] = {
                "site_id": "atop:0",
                "state_dir": "03_OOH",
                "species": "OOH",
                "candidates_traj": str(ooh_dir / "candidates.traj"),
                "candidates_csv": str(ooh_dir / "candidates.csv"),
            }
            cfg = SimpleNamespace(
                surface_side="top",
                candidate_generation={"ooh_orientations": 6},
                local_cmc={
                    "enabled": True,
                    "states": ["02_O", "03_OOH"],
                    "sequential": {
                        "enabled": True,
                        "rules": [
                            {
                                "source_state": "02_O",
                                "target_state": "03_OOH",
                                "builder": "ooh_from_o",
                                "n_orientations": 1,
                            }
                        ],
                    },
                    "progress_stdout": False,
                },
            )

            workflow = ReactionLocalCMCWorkflow(cfg)
            rules = workflow._sequential_rules({"02_O", "03_OOH"})
            workflow._refresh_sequential_target_candidates(
                ooh_row,
                {("atop:0", "02_O"): o_row, ("atop:0", "03_OOH"): ooh_row},
                rules,
            )

            with (ooh_dir / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row["parent_o_candidate_id"] for row in rows],
                ["o_basin_a", "o_basin_b"],
            )
            self.assertEqual(
                [row["parent_state_candidate_id"] for row in rows],
                ["o_basin_a", "o_basin_b"],
            )
            self.assertEqual(rows[0]["parent_state_dir"], "02_O")
            self.assertEqual(rows[0]["transition_builder"], "ooh_from_o")
            self.assertEqual(ooh_row["n_candidates"], 2)
            self.assertTrue(ooh_row["_local_cmc_use_all_seeds"])
            self.assertTrue(ooh_row["_local_cmc_ignore_existing"])

    def test_reaction_local_cmc_compatible_routes_enable_default_oer_sequence(self):
        cfg = SimpleNamespace(
            che={"route_pairing_mode": "compatible"},
            local_cmc={
                "enabled": True,
                "states": ["02_O", "03_OOH"],
                "sequential_ooh_from_o": False,
            },
        )
        rules = ReactionLocalCMCWorkflow(cfg)._sequential_rules({"02_O", "03_OOH"})

        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["source_state"], "02_O")
        self.assertEqual(rules[0]["target_state"], "03_OOH")
        self.assertEqual(rules[0]["builder"], "ooh_from_o")

    def test_reaction_local_cmc_generic_sequential_rules_order_chained_states(self):
        rows = [
            {"site_id": "site0", "state_dir": "C"},
            {"site_id": "site0", "state_dir": "A"},
            {"site_id": "site0", "state_dir": "B"},
        ]
        rules = [
            {"source_state": "B", "target_state": "C", "builder": "noop"},
            {"source_state": "A", "target_state": "B", "builder": "noop"},
        ]

        ordered = ReactionLocalCMCWorkflow._local_work_order(
            rows,
            {"A", "B", "C"},
            rules,
        )

        self.assertEqual([row["state_dir"] for row in ordered], ["A", "B", "C"])

    def test_reaction_local_cmc_passes_puckering_controls_to_driver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_dir = root / "local_cmc"
            local_dir.mkdir()
            candidate_path = root / "sites" / "atop_0" / "reactions" / "03_OOH" / "candidates.traj"
            candidate_path.parent.mkdir(parents=True)
            atoms = Atoms(
                "PtOOH",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.8],
                    [0.0, 0.0, 2.9],
                    [0.0, 0.0, 3.8],
                ],
                cell=[8.0, 8.0, 12.0],
                pbc=[False, False, False],
            )
            atoms.set_tags([0, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET, ADSORBATE_TAG_OFFSET])
            captured = {}

            class RecordingCMC:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.atoms = kwargs["atoms"].copy()

                def run(self, nsweeps, traj_file, **kwargs):
                    write(traj_file, [self.atoms])

            cfg = SimpleNamespace(
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                local_cmc={
                    "enabled": True,
                    "move_mode": "hybrid",
                    "site_hop_prob": 0.3,
                    "reorientation_prob": 0.2,
                    "puckering_prob": 0.4,
                    "puckering_hop_prob": 0.1,
                    "puckering_elements": ["Pt"],
                    "puckering_height_A": 0.7,
                    "max_puckering_trials": 9,
                    "progress_stdout": False,
                },
            )

            with patch("reaction.local_cmc.AdsorbateCMC", RecordingCMC):
                ReactionLocalCMCWorkflow(cfg)._run_seed(
                    atoms,
                    {"candidate_id": "ooh0"},
                    {"candidates_traj": str(candidate_path)},
                    local_dir,
                    0,
                    calculator=None,
                )

            self.assertEqual(captured["puckering_prob"], 0.4)
            self.assertEqual(captured["puckering_hop_prob"], 0.1)
            self.assertEqual(captured["puckering_elements"], ("Pt",))
            self.assertEqual(captured["puckering_height_A"], 0.7)
            self.assertEqual(captured["max_puckering_trials"], 9)

    def test_reaction_local_cmc_accepts_structured_config_sections(self):
        cfg = SimpleNamespace(
            local_cmc={
                "enabled": True,
                "states": ["03_OOH"],
                "region": {
                    "center_state": "01_OH",
                    "radius_A": 3.5,
                    "distance_metric": "xy",
                },
                "sampling": {
                    "temperature_K": 298.0,
                    "n_cycles": 80,
                    "sample_interval": 4,
                    "max_seed_candidates_per_state": 2,
                },
                "pt": {
                    "enabled": True,
                    "temperatures_K": [298.0, 400.0],
                    "target_temperature_K": 298.0,
                    "swap_interval": 8,
                    "local_eq_fraction": 0.1,
                },
                "backend": {
                    "backend": "ray",
                    "n_gpus": 2,
                    "workers_per_gpu": 1,
                    "ray_num_gpus_per_task": 1.0,
                },
                "moves": {
                    "mode": "hybrid",
                    "site_hop_prob": 0.25,
                    "reorientation_prob": 0.1,
                    "puckering": {
                        "prob": 0.2,
                        "hop_prob": 0.1,
                        "elements": ["Pt"],
                        "height_A": 0.15,
                    },
                },
                "relaxation": {"enabled": True, "steps": 30, "fmax": 0.04},
                "md": {"enabled": False, "move_prob": 0.0},
                "output": {"write_debug_trajs": True, "progress_stdout": False},
            }
        )

        local = ReactionLocalCMCWorkflow(cfg).local_config

        self.assertTrue(local["pt_enabled"])
        self.assertEqual(local["temperatures_K"], [298.0, 400.0])
        self.assertEqual(local["backend"], "ray")
        self.assertEqual(local["n_gpus"], 2)
        self.assertEqual(local["move_mode"], "hybrid")
        self.assertEqual(local["puckering_prob"], 0.2)
        self.assertEqual(local["puckering_hop_prob"], 0.1)
        self.assertEqual(local["puckering_elements"], ["Pt"])
        self.assertTrue(local["relax"])
        self.assertFalse(local["enable_hybrid_md"])
        self.assertTrue(local["write_debug_trajs"])

    def test_reaction_local_cmc_diverse_output_selection_keeps_late_basin(self):
        cfg = SimpleNamespace(
            local_cmc={
                "output_selection": "diverse",
                "max_output_candidates_per_state": 3,
            }
        )
        workflow = ReactionLocalCMCWorkflow(cfg)
        samples = []
        for index in range(20):
            if index == 15:
                ads_pos = [1.4, 1.4, 1.8]
            else:
                ads_pos = [0.02 * index, 0.0, 1.8]
            atoms = Atoms(
                "PtO",
                positions=[[0.0, 0.0, 0.0], ads_pos],
                cell=[8.0, 8.0, 12.0],
                pbc=[False, False, False],
            )
            atoms.set_tags([0, ADSORBATE_TAG_OFFSET])
            samples.append((atoms, {"sample": index}))

        selected = workflow._select_output_samples(samples, 3)

        selected_indices = [metadata["sample"] for _, metadata in selected]
        self.assertIn(15, selected_indices)
        self.assertNotEqual(selected_indices, [0, 1, 2])

    def test_reaction_local_cmc_motif_diverse_output_selection_keeps_rare_atop(self):
        cfg = SimpleNamespace(
            site_elements=("Pt",),
            substrate_elements=("Pt",),
            local_cmc={
                "output_selection": "motif_diverse",
                "max_output_candidates_per_state": 3,
                "motif_atop_distance_A": 0.6,
            },
        )
        workflow = ReactionLocalCMCWorkflow(cfg)
        samples = []
        support = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
        for index in range(20):
            ads_pos = [1.6 + 0.01 * index, 1.6, 1.8]
            if index == 17:
                ads_pos = [0.2, 0.0, 1.8]
            atoms = Atoms(
                "Pt3O",
                positions=[*support, ads_pos],
                cell=[8.0, 8.0, 12.0],
                pbc=[False, False, False],
            )
            atoms.set_tags([0, 0, 0, ADSORBATE_TAG_OFFSET])
            samples.append((atoms, {"sample": index}))

        selected = workflow._select_output_samples(samples, 3)

        selected_indices = [metadata["sample"] for _, metadata in selected]
        self.assertIn(17, selected_indices)
        self.assertNotEqual(selected_indices, [0, 1, 2])

    def test_reaction_local_cmc_first_output_selection_keeps_legacy_order(self):
        cfg = SimpleNamespace(local_cmc={"output_selection": "first"})
        workflow = ReactionLocalCMCWorkflow(cfg)
        samples = [
            (Atoms("H", positions=[[float(index), 0.0, 0.0]]), {"sample": index})
            for index in range(5)
        ]

        selected = workflow._select_output_samples(samples, 3)

        self.assertEqual([metadata["sample"] for _, metadata in selected], [0, 1, 2])

    def test_reaction_local_cmc_skip_existing_repromotes_pt_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            site_dir = root / "sites" / "atop_0"
            state_dir = site_dir / "reactions" / "02_O"
            local_dir = state_dir / "local_cmc"
            pt_dir = local_dir / "seed000_pt"
            pt_dir.mkdir(parents=True)
            local_dir.mkdir(exist_ok=True)

            seed = Atoms(
                "PtO",
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
                cell=[8.0, 8.0, 12.0],
                pbc=[False, False, False],
            )
            seed.set_tags([0, ADSORBATE_TAG_OFFSET])
            write(str(state_dir / "candidates.traj"), [seed])
            (state_dir / "candidates.csv").write_text(
                "site_id,state,species,candidate_id,candidate_kind,anchor_index,"
                "source_representative_rank\n"
                "atop:0,02_O,O,o_seed,seed,1,0\n"
            )
            frames = []
            for index in range(5):
                frame = seed.copy()
                frame.positions[1, 0] = float(index)
                frames.append(frame)
            write(str(pt_dir / "replica_300K.traj"), frames)
            (local_dir / "done").write_text("existing samples=5\n")

            manifest = root / "candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "site_id": "atop:0",
                        "site_dir": str(site_dir),
                        "state_dir": "02_O",
                        "species": "O",
                        "n_candidates": 1,
                        "candidates_traj": str(state_dir / "candidates.traj"),
                        "candidates_csv": str(state_dir / "candidates.csv"),
                    }
                )

            cfg = SimpleNamespace(
                output_dir=str(root),
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                local_cmc={
                    "enabled": True,
                    "states": ["02_O"],
                    "pt_enabled": True,
                    "temperatures_K": [300.0],
                    "target_temperature_K": 300.0,
                    "skip_existing": True,
                    "max_output_candidates_per_state": 2,
                    "output_selection": "last",
                    "progress_stdout": False,
                },
            )

            ReactionLocalCMCWorkflow(cfg).run(manifest)

            promoted = read(str(state_dir / "candidates.traj"), ":")
            with (state_dir / "candidates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(promoted), 2)
            self.assertEqual(rows[0]["candidate_id"], "o_seed_localpt003")
            self.assertEqual(rows[1]["candidate_id"], "o_seed_localpt004")
            self.assertAlmostEqual(promoted[0].positions[1, 0], 3.0)
            self.assertAlmostEqual(promoted[1].positions[1, 0], 4.0)

    def test_reaction_local_cmc_pt_uses_replica_exchange(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            local_dir = root / "local_cmc"
            local_dir.mkdir()
            candidate_path = (
                root
                / "sites"
                / "atop_0"
                / "reactions"
                / "02_O"
                / "candidates.traj"
            )
            candidate_path.parent.mkdir(parents=True)
            atoms = Atoms(
                "PtO",
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]],
                cell=[8.0, 8.0, 12.0],
                pbc=[False, False, False],
            )
            atoms.set_tags([0, ADSORBATE_TAG_OFFSET])
            captured = {}

            class RecordingPT:
                def __init__(self, **kwargs):
                    captured.update(kwargs)
                    self.replica_states = kwargs["replica_states"]

                def run(self, n_cycles, equilibration_cycles=0):
                    captured["n_cycles"] = n_cycles
                    captured["equilibration_cycles"] = equilibration_cycles
                    for state in self.replica_states:
                        write(state["traj_file"], [state["atoms"]])

            cfg = SimpleNamespace(
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                site_elements=("Pt",),
                substrate_elements=("Pt",),
                functional_elements=(),
                site_types=("atop",),
                surface_side="top",
                surface_layer_tol=0.5,
                site_match_tol=0.6,
                support_xy_tol=1.2,
                termination_site_xy_tol=2.2,
                vertical_offset=1.8,
                termination_clearance=0.0,
                local_cmc={
                    "enabled": True,
                    "pt_enabled": True,
                    "temperatures_K": [300.0, 450.0],
                    "temperature_K": 300.0,
                    "n_cycles": 20,
                    "swap_interval": 5,
                    "sample_interval": 1,
                    "equilibration_cycles": 0,
                    "backend": "ray",
                    "n_gpus": 1,
                    "workers_per_gpu": 1,
                    "ray_num_gpus_per_task": 1.0,
                    "progress_stdout": False,
                },
            )

            with patch("reaction.local_cmc.ReplicaExchange", RecordingPT):
                outputs = ReactionLocalCMCWorkflow(cfg)._run_seed(
                    atoms,
                    {"candidate_id": "o0"},
                    {"candidates_traj": str(candidate_path)},
                    local_dir,
                    0,
                    calculator=None,
                )

            self.assertEqual(captured["execution_backend"], "ray")
            self.assertEqual(captured["n_gpus"], 1)
            self.assertEqual(captured["workers_per_gpu"], 1)
            self.assertEqual(captured["n_cycles"], 4)
            self.assertEqual(len(captured["replica_states"]), 2)
            self.assertIn("seed000_pt", captured["replica_states"][0]["traj_file"])
            self.assertEqual(outputs[0][1]["candidate_kind"], "local_cmc_pt_sample")
            self.assertEqual(outputs[0][1]["local_cmc_pt_temperature_K"], 300.0)

    def test_state_relaxer_writes_relaxed_state_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_dir = root / "sites" / "fcc_1-2-3" / "reactions" / "02_O"
            state_dir.mkdir(parents=True)
            candidates_traj = state_dir / "candidates.traj"
            candidates_csv = state_dir / "candidates.csv"
            write(
                str(candidates_traj),
                [Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])],
            )
            candidates_csv.write_text(
                "site_id,state,species,candidate_id,candidate_kind,anchor_index\n"
                "fcc:1-2-3,02_O,O,test_o,direct_deprotonation,0\n"
            )
            manifest = root / "oh_parent_sites_candidate_manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "site_id",
                        "site_dir",
                        "state_dir",
                        "species",
                        "n_candidates",
                        "candidates_traj",
                        "candidates_csv",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "site_id": "fcc:1-2-3",
                        "site_dir": str(state_dir.parents[1]),
                        "state_dir": "02_O",
                        "species": "O",
                        "n_candidates": 1,
                        "candidates_traj": str(candidates_traj),
                        "candidates_csv": str(candidates_csv),
                    }
                )

            cfg = SimpleNamespace(
                output_dir=str(root),
                out_prefix="oh_parent_sites",
                calculator="lj",
                lj_cutoff=6.0,
                device="cpu",
                model=None,
                model_file=None,
                use_kokkos=True,
                state_relaxation={
                    "enabled": True,
                    "states": ["02_O"],
                    "steps": 1,
                    "fmax": 10.0,
                    "progress_stdout": False,
                },
            )

            outputs = ReactionStateRelaxer(cfg).relax(manifest)

            self.assertEqual(
                outputs["state_relaxation_manifest_csv"],
                str(root / "state_relaxation_manifest.csv"),
            )
            self.assertTrue((state_dir / "relaxed.traj").exists())
            self.assertTrue((state_dir / "energies.csv").exists())
            progress_log = root / "state_relaxation.log"
            self.assertTrue(progress_log.exists())
            self.assertIn("candidate 1/1 done", progress_log.read_text())
            relaxed = read(str(state_dir / "relaxed.traj"), index=":")
            self.assertEqual(len(relaxed), 1)
            self.assertIn("reaction_relaxed_energy_eV", relaxed[0].info)

    def test_state_relaxer_detects_dissociated_molecular_adsorbate(self):
        cfg = SimpleNamespace(
            state_relaxation={
                "enabled": True,
                "enforce_adsorbate_integrity": True,
            },
        )
        relaxer = ReactionStateRelaxer(cfg)
        initial = Atoms(
            "OOH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.45, 0.0, 0.0],
                [2.43, 0.0, 0.0],
            ],
        )
        initial.new_array("reaction_is_adsorbate", np.ones(3, dtype=int))
        relaxed = initial.copy()
        relaxed.positions[1] = [3.5, 0.0, 0.0]

        intact, error, distances = relaxer._adsorbate_integrity(
            initial,
            relaxed,
            {},
            {"species": "OOH"},
        )

        self.assertFalse(intact)
        self.assertIn("AdsorbateIntegrityError", error)
        self.assertIn("0-1", distances)

    def test_oer_che_summarizer_excludes_nonintact_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_dir = root / "sites" / "fcc_1-2-3" / "reactions" / "03_OOH"
            state_dir.mkdir(parents=True)
            (state_dir / "candidates.csv").write_text("candidate_id\nbad\ngood\n")
            (state_dir / "energies.csv").write_text(
                "candidate_id,converged,energy_eV,adsorbate_intact,error\n"
                "bad,True,-10.0,False,AdsorbateIntegrityError\n"
                "good,True,-5.0,True,\n"
            )
            cfg = SimpleNamespace(output_dir=str(root), che={"enabled": True})
            summarizer = OERCHESummarizer(cfg)

            _, finite, _ = summarizer._finite_energy_rows(
                {"candidates_csv": str(state_dir / "candidates.csv")}
            )

        self.assertEqual([row["candidate_id"] for row in finite], ["good"])

    def test_vibration_masks_reuse_clean_slab_origins(self):
        cfg = SimpleNamespace(
            output_dir="/tmp/reaction-vib-test",
            vibrations={
                "enabled": True,
                "slab_cutoff_A": 0.8,
                "progress_stdout": False,
            },
        )
        workflow = ReactionStateVibrationWorkflow(cfg)

        clean = Atoms(
            "Pt4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
                [3.0, 3.0, 0.0],
            ],
        )
        clean.new_array("reaction_origin_index", np.asarray([0, 1, 2, 3], dtype=int))
        clean.new_array("reaction_is_adsorbate", np.asarray([0, 0, 0, 0], dtype=int))

        slab_origins = workflow._clean_local_slab_origins("fcc:0-1-2", clean)
        self.assertEqual(slab_origins, (0, 1, 2))

        oh = Atoms(
            "Pt4OH",
            positions=[
                [0.5, 0.8, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.4, 1.4],
                [3.0, 3.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.4, 2.3],
            ],
        )
        oh.new_array("reaction_origin_index", np.asarray([2, 0, -1, 3, 1, -2], dtype=int))
        oh.new_array("reaction_is_adsorbate", np.asarray([0, 0, 1, 0, 0, 1], dtype=int))

        vib_indices, slab_indices, adsorbate_indices = workflow._state_vibration_indices(
            oh,
            "oh",
            slab_origins,
        )

        self.assertEqual({int(oh.arrays["reaction_origin_index"][idx]) for idx in slab_indices}, {0, 1, 2})
        self.assertEqual(adsorbate_indices, [2, 5])
        self.assertEqual(vib_indices, [0, 1, 2, 4, 5])

    def test_vibration_threshold_policy_accepts_soft_imaginary_modes(self):
        workflow = ReactionStateVibrationWorkflow(
            SimpleNamespace(
                vibrations={
                    "imag_mode_policy": "threshold",
                    "imag_frequency_threshold_cm1": 30.0,
                    "max_imag_modes": 1,
                }
            )
        )
        vib_energies = [0.10, 1j * 20.0 * units.invcm, 0.20]
        summary = workflow._imaginary_mode_summary(vib_energies)

        cleaned = workflow._thermo_vib_energies(vib_energies, summary)

        self.assertEqual(summary["n_imag_modes"], 1)
        self.assertAlmostEqual(summary["max_imag_frequency_cm1"], 20.0)
        self.assertEqual(summary["status"], "accepted_soft")
        self.assertEqual(list(cleaned), [0.10, 0.20])

    def test_vibration_threshold_policy_rejects_large_imaginary_modes(self):
        workflow = ReactionStateVibrationWorkflow(
            SimpleNamespace(
                vibrations={
                    "imag_mode_policy": "threshold",
                    "imag_frequency_threshold_cm1": 30.0,
                    "max_imag_modes": 1,
                }
            )
        )
        vib_energies = [0.10, 1j * 80.0 * units.invcm, 0.20]
        summary = workflow._imaginary_mode_summary(vib_energies)

        with self.assertRaises(ValueError):
            workflow._thermo_vib_energies(vib_energies, summary)
        self.assertEqual(summary["status"], "rejected")

    def test_reference_thermo_che_snippet_uses_expected_keys(self):
        summary_h2 = {
            "reference_name": "H2",
            "potential_energy_eV": -6.0,
            "temperature_K": 303.0,
            "pressure_Pa": 101325.0,
            "geometry": "linear",
            "symmetrynumber": 2,
            "spin": 0.0,
            "vib_energies_eV": [0.5],
            "correction_eV": 0.02,
        }
        snippet_h2 = ReferenceThermoWorkflow._che_snippet(summary_h2)
        self.assertIn("h2_energy_eV", snippet_h2["che"])
        self.assertIn("h2_thermo", snippet_h2["che"])
        self.assertEqual(snippet_h2["che"]["h2_correction_eV"], 0.02)

        summary_h2o = dict(summary_h2)
        summary_h2o["reference_name"] = "H2O"
        summary_h2o["geometry"] = "nonlinear"
        snippet_h2o = ReferenceThermoWorkflow._che_snippet(summary_h2o)
        self.assertIn("h2o_energy_eV", snippet_h2o["che"])
        self.assertIn("h2o_thermo", snippet_h2o["che"])

        summary_o2 = dict(summary_h2)
        summary_o2["reference_name"] = "O2"
        summary_o2["spin"] = 1.0
        snippet_o2 = ReferenceThermoWorkflow._che_snippet(summary_o2)
        self.assertIn("o2_energy_eV", snippet_o2["che"])
        self.assertIn("o2_thermo", snippet_o2["che"])

    def test_reference_thermo_uses_configured_energy_override(self):
        workflow = ReferenceThermoWorkflow(SimpleNamespace(energy_eV=-6.98))
        energy, source = workflow._thermo_reference_energy(calculated_energy=-1.0)

        self.assertEqual(energy, -6.98)
        self.assertEqual(source, "configured")

    def test_reference_thermo_stage_updates_main_che_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = SimpleNamespace(
                output_dir=str(root),
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                che={"route_temperature_K": 303.0},
                reference_thermo={
                    "enabled": True,
                    "molecules": ["H2", "H2O"],
                    "output_dir": "references",
                    "skip_existing": True,
                },
            )

            def fake_run(workflow):
                output_dir = Path(str(workflow.config.output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                label = str(workflow.config.molecule).lower()
                snippet = {
                    "che": {
                        f"{label}_energy_eV": -1.0,
                        f"{label}_thermo": {
                            "enabled": True,
                            "vib_energies_eV": [0.1],
                        },
                    }
                }
                snippet_path = output_dir / "che_snippet.yaml"
                snippet_path.write_text(yaml.safe_dump(snippet))
                return {
                    "initial_traj": str(output_dir / "initial.traj"),
                    "relaxed_traj": str(output_dir / "relaxed.traj"),
                    "summary_yaml": str(output_dir / "summary.yaml"),
                    "che_snippet_yaml": str(snippet_path),
                }

            with patch.object(ReferenceThermoWorkflow, "run", fake_run):
                outputs = run_reference_thermo_stage(cfg)

        self.assertEqual(cfg.che["h2_energy_eV"], -1.0)
        self.assertEqual(cfg.che["h2o_energy_eV"], -1.0)
        self.assertTrue(cfg.che["h2_thermo"]["enabled"])
        self.assertTrue(cfg.che["h2o_thermo"]["enabled"])
        self.assertIn("reference_thermo_h2_che_snippet_yaml", outputs)
        self.assertIn("reference_thermo_h2o_che_snippet_yaml", outputs)

    def test_reference_thermo_stage_energy_override_replaces_manual_che_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = SimpleNamespace(
                output_dir=str(root),
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                che={
                    "h2_energy_eV": -99.0,
                    "h2_thermo": {"enabled": False},
                },
                reference_thermo={
                    "enabled": True,
                    "molecules": ["H2"],
                    "output_dir": "references",
                    "skip_existing": True,
                    "h2": {"energy_eV": -6.98},
                },
            )

            def fake_run(workflow):
                self.assertEqual(workflow.config.energy_eV, -6.98)
                output_dir = Path(str(workflow.config.output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                snippet = {
                    "che": {
                        "h2_energy_eV": workflow.config.energy_eV,
                        "h2_thermo": {
                            "enabled": True,
                            "vib_energies_eV": [0.1],
                        },
                    }
                }
                snippet_path = output_dir / "che_snippet.yaml"
                snippet_path.write_text(yaml.safe_dump(snippet))
                return {
                    "initial_traj": str(output_dir / "initial.traj"),
                    "relaxed_traj": str(output_dir / "relaxed.traj"),
                    "summary_yaml": str(output_dir / "summary.yaml"),
                    "che_snippet_yaml": str(snippet_path),
                }

            with patch.object(ReferenceThermoWorkflow, "run", fake_run):
                run_reference_thermo_stage(cfg)

        self.assertEqual(cfg.che["h2_energy_eV"], -6.98)
        self.assertTrue(cfg.che["h2_thermo"]["enabled"])

    def test_reference_thermo_stage_does_not_overwrite_manual_che_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = SimpleNamespace(
                output_dir=str(root),
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                che={
                    "h2_energy_eV": -99.0,
                    "h2_thermo": {"enabled": False},
                },
                reference_thermo={
                    "enabled": True,
                    "molecules": ["H2", "H2O"],
                    "output_dir": "references",
                    "skip_existing": True,
                },
            )

            def fake_run(workflow):
                output_dir = Path(str(workflow.config.output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                label = str(workflow.config.molecule).lower()
                snippet = {
                    "che": {
                        f"{label}_energy_eV": -1.0,
                        f"{label}_thermo": {
                            "enabled": True,
                            "vib_energies_eV": [0.1],
                        },
                    }
                }
                snippet_path = output_dir / "che_snippet.yaml"
                snippet_path.write_text(yaml.safe_dump(snippet))
                return {
                    "initial_traj": str(output_dir / "initial.traj"),
                    "relaxed_traj": str(output_dir / "relaxed.traj"),
                    "summary_yaml": str(output_dir / "summary.yaml"),
                    "che_snippet_yaml": str(snippet_path),
                }

            with patch.object(ReferenceThermoWorkflow, "run", fake_run):
                run_reference_thermo_stage(cfg)

        self.assertEqual(cfg.che["h2_energy_eV"], -99.0)
        self.assertFalse(cfg.che["h2_thermo"]["enabled"])
        self.assertEqual(cfg.che["h2o_energy_eV"], -1.0)
        self.assertTrue(cfg.che["h2o_thermo"]["enabled"])

    def test_reference_thermo_stage_can_overwrite_manual_che_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = SimpleNamespace(
                output_dir=str(root),
                calculator="lj",
                lj_cutoff=6.0,
                model=None,
                model_file=None,
                device="cpu",
                use_kokkos=True,
                che={
                    "h2_energy_eV": -99.0,
                    "h2_thermo": {"enabled": False},
                },
                reference_thermo={
                    "enabled": True,
                    "molecules": ["H2"],
                    "output_dir": "references",
                    "skip_existing": True,
                    "overwrite_che_references": True,
                },
            )

            def fake_run(workflow):
                output_dir = Path(str(workflow.config.output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                snippet = {
                    "che": {
                        "h2_energy_eV": -1.0,
                        "h2_thermo": {
                            "enabled": True,
                            "vib_energies_eV": [0.1],
                        },
                    }
                }
                snippet_path = output_dir / "che_snippet.yaml"
                snippet_path.write_text(yaml.safe_dump(snippet))
                return {
                    "initial_traj": str(output_dir / "initial.traj"),
                    "relaxed_traj": str(output_dir / "relaxed.traj"),
                    "summary_yaml": str(output_dir / "summary.yaml"),
                    "che_snippet_yaml": str(snippet_path),
                }

            with patch.object(ReferenceThermoWorkflow, "run", fake_run):
                run_reference_thermo_stage(cfg)

        self.assertEqual(cfg.che["h2_energy_eV"], -1.0)
        self.assertTrue(cfg.che["h2_thermo"]["enabled"])


if __name__ == "__main__":
    unittest.main()
