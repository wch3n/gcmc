import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.thermochemistry import IdealGasThermo
from reaction import (
    OERCHESummarizer,
    ReferenceThermoWorkflow,
    ReactionCandidateGenerator,
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
            self.assertEqual(
                list(rows[0].keys()),
                [
                    "site_id",
                    "site_population_rank",
                    "population_total",
                    "clean_reference_site_id",
                    "ready_min",
                    "DeltaG1_min_eV",
                    "DeltaG2_min_eV",
                    "DeltaG3_min_eV",
                    "DeltaG4_min_eV",
                    "ready_min_electronic",
                    "DeltaG1_min_electronic_eV",
                    "DeltaG2_min_electronic_eV",
                    "DeltaG3_min_electronic_eV",
                    "DeltaG4_min_electronic_eV",
                    "limiting_step_min_electronic",
                    "limiting_DeltaG_min_electronic_eV",
                    "overpotential_min_electronic_V",
                    "limiting_step_min",
                    "limiting_DeltaG_min_eV",
                    "overpotential_min_V",
                    "missing_states_min",
                    "missing_states_min_electronic",
                    "ready_boltzmann",
                    "boltzmann_temperature_K",
                    "boltzmann_energy_cluster_tol_eV",
                    "n_OH_candidates",
                    "n_O_candidates",
                    "n_OOH_candidates",
                    "DeltaG1_boltzmann_eV",
                    "DeltaG2_boltzmann_eV",
                    "DeltaG3_boltzmann_eV",
                    "DeltaG4_boltzmann_eV",
                    "ready_boltzmann_electronic",
                    "DeltaG1_boltzmann_electronic_eV",
                    "DeltaG2_boltzmann_electronic_eV",
                    "DeltaG3_boltzmann_electronic_eV",
                    "DeltaG4_boltzmann_electronic_eV",
                    "limiting_step_boltzmann_electronic",
                    "limiting_DeltaG_boltzmann_electronic_eV",
                    "overpotential_boltzmann_electronic_V",
                    "limiting_step_boltzmann",
                    "limiting_DeltaG_boltzmann_eV",
                    "overpotential_boltzmann_V",
                    "missing_states_boltzmann",
                    "missing_states_boltzmann_electronic",
                    "potential_V",
                    "limiting_DeltaG_min_at_U_eV",
                    "limiting_DeltaG_min_electronic_at_U_eV",
                    "limiting_DeltaG_boltzmann_at_U_eV",
                    "limiting_DeltaG_boltzmann_electronic_at_U_eV",
                ],
            )
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

    def test_oer_che_summarizer_writes_boltzmann_weighted_summary(self):
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
                    "boltzmann_weight_states": True,
                    "boltzmann_temperature_K": 300.0,
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
            kbt = 8.617333262145e-5 * 300.0
            self.assertEqual(rows[0]["n_O_candidates"], "2")
            self.assertAlmostEqual(
                float(rows[0]["DeltaG2_boltzmann_eV"]),
                1.0 - kbt * math.log(2.0),
            )
            self.assertAlmostEqual(
                float(rows[0]["DeltaG2_boltzmann_electronic_eV"]),
                1.0 - kbt * math.log(2.0),
            )
            with Path(outputs["oer_ensemble_csv"]).open() as handle:
                ensemble = list(csv.DictReader(handle))
            self.assertEqual(
                list(ensemble[0].keys()),
                [
                    "source",
                    "n_ready_sites",
                    "population_sum_ready",
                    "population_missing",
                    "site_weight_model",
                    "site_degeneracy_model",
                    "boltzmann_temperature_K",
                    "boltzmann_energy_cluster_tol_eV",
                    "route_weighted_mean_overpotential_V",
                    "min_site_overpotential_V",
                    "min_site_overpotential_site_id",
                    "dominant_weight_site_id",
                    "dominant_weight_fraction",
                ],
            )
            self.assertAlmostEqual(
                float(ensemble[0]["route_weighted_mean_overpotential_V"]),
                0.77,
            )
            self.assertEqual(ensemble[0]["site_weight_model"], "dilute_oh_deltag1")

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
                che={"boltzmann_temperature_K": 303.0},
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
