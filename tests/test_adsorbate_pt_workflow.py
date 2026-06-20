import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ase import Atoms
from ase.io import read

from gcmc.adsorbate_cmc import AdsorbateCMC
from gcmc.constants import ADSORBATE_TAG_OFFSET
from gcmc.workflows import (
    AdsorbateReplicaExchangeWorkflow,
    _DEFAULT_ADSORBATE_PT_CONFIG,
    _select_initial_adsorbate_template,
    load_adsorbate_pt_config,
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


class FakeReplicaExchange:
    def __init__(self):
        self.replica_states = [
            {
                "traj_file": "replica_400K.traj",
                "thermo_file": "replica_400K.dat",
                "checkpoint_file": "checkpoint_400K.pkl",
            },
            {
                "traj_file": "replica_300K.traj",
                "thermo_file": "replica_300K.dat",
                "checkpoint_file": "checkpoint_300K.pkl",
            },
        ]
        self.run_args = None

    def run(self, n_cycles, equilibration_cycles=0):
        self.run_args = (n_cycles, equilibration_cycles)


class AdsorbatePTWorkflowTests(unittest.TestCase):
    def test_load_adsorbate_pt_config_resolves_output_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            template_dir = Path(tmpdir) / "templates"
            template_dir.mkdir()
            (template_dir / "OOH_flat.vasp").write_text(
                """
O H
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
O H
2 1
Cartesian
0.0 0.0 0.0
1.2 0.0 0.0
1.3 0.8 0.1
""".strip()
            )
            config_path.write_text(
                """
system:
  snapshot: slab.traj
  adsorbate_templates:
    - name: upright
      adsorbate: OOH
      weight: 1.0
      anchor:
        mode: atom
        atom_index: 0
    - name: flat
      path: templates/OOH_flat.vasp
      weight: 2.0
      anchor:
        mode: center_of_mass
        atom_indices: [0, 1]
pt:
  T_start: 500
  T_end: 300
  T_step: 100
cmc:
  moves:
    mode: hybrid
    orientation_filter:
      atom_indices: [2]
      min_z_above_anchor_A: 0.1
    diagnostics:
      enabled: true
      log: true
      top_n: 2
    hop:
      prob: 0.6
      reorient:
        prob: 0.75
        angle_deg: 160
        max_trials: 7
      puckering:
        prob: 0.5
        elements: [Pt]
        height_A: 0.3
        height_jitter_A: 0.02
output:
  output_dir: results
  stats_file: stats.csv
  results_file: results.csv
  checkpoint_file: state.pkl
  initial_traj_file: init.traj
  debug_traj_interval: 25
""".strip()
            )

            cfg = load_adsorbate_pt_config(config_path)
            selected = _select_initial_adsorbate_template(cfg, seed=1)

        root = (Path(tmpdir) / "results").resolve()
        self.assertEqual(cfg.output_dir, str(root))
        self.assertEqual(cfg.stats_file, str(root / "stats.csv"))
        self.assertEqual(cfg.results_file, str(root / "results.csv"))
        self.assertEqual(cfg.checkpoint_file, str(root / "state.pkl"))
        self.assertEqual(cfg.initial_traj_file, str(root / "init.traj"))
        self.assertEqual(cfg.adsorbate_templates[0]["adsorbate"], "OOH")
        self.assertEqual(
            cfg.adsorbate_templates[1]["path"],
            str((Path(tmpdir) / "templates" / "OOH_flat.vasp").resolve()),
        )
        self.assertEqual(cfg.adsorbate_templates[1]["anchor"]["mode"], "center_of_mass")
        self.assertEqual(cfg.debug_traj_interval, 25)
        self.assertEqual(cfg.move_mode, "hybrid")
        self.assertAlmostEqual(cfg.site_hop_prob, 0.075)
        self.assertAlmostEqual(cfg.hop_reorientation_prob, 0.225)
        self.assertAlmostEqual(cfg.hop_puckering_prob, 0.075)
        self.assertAlmostEqual(cfg.hop_puckering_reorientation_prob, 0.225)
        self.assertEqual(cfg.hop_reorientation_angle_deg, 160)
        self.assertEqual(cfg.max_hop_reorientation_trials, 7)
        self.assertEqual(cfg.puckering_elements, ["Pt"])
        self.assertEqual(cfg.puckering_height_A, 0.3)
        self.assertEqual(cfg.puckering_height_jitter_A, 0.02)
        self.assertEqual(cfg.molecular_upright_atom_indices, [2])
        self.assertEqual(cfg.molecular_upright_min_z_A, 0.1)
        self.assertTrue(cfg.diagnostics_enabled)
        self.assertTrue(cfg.diagnostics_log)
        self.assertEqual(cfg.diagnostics_top_n, 2)
        template, anchor_index, anchor_mode, anchor_atom_indices, library = selected
        self.assertEqual(template.get_chemical_formula(), "HO2")
        self.assertEqual(anchor_index, 0)
        self.assertEqual(anchor_mode, "center_of_mass")
        self.assertEqual(anchor_atom_indices, [0, 1])
        self.assertEqual(len(library), 2)

    def test_adsorbate_pt_workflow_initializes_fixed_count_and_relocates_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_dict = dict(_DEFAULT_ADSORBATE_PT_CONFIG)
            cfg_dict.update(
                {
                    "snapshot": str(Path(tmpdir) / "dummy.traj"),
                    "frame": 0,
                    "adsorbate": "OH",
                    "initialization_mode": "fixed_count",
                    "n_adsorbates": 1,
                    "site_elements": ["Ti", "Zr", "Mo"],
                    "surface_side": "top",
                    "site_type": ["atop"],
                    "substrate_elements": ["Ti", "Zr", "Mo", "C"],
                    "functional_elements": ["O"],
                    "vertical_offset": 1.5,
                    "T_start": 400.0,
                    "T_end": 300.0,
                    "T_step": 100.0,
                    "n_cycles": 3,
                    "equilibration_cycles": 1,
                    "backend": "multiprocessing",
                    "n_gpus": 1,
                    "workers_per_gpu": 1,
                    "output_dir": str(Path(tmpdir) / "pt_out"),
                    "stats_file": str(Path(tmpdir) / "pt_out" / "stats.csv"),
                    "results_file": str(Path(tmpdir) / "pt_out" / "results.csv"),
                    "checkpoint_file": str(Path(tmpdir) / "pt_out" / "state.pkl"),
                    "initial_traj_file": str(Path(tmpdir) / "pt_out" / "initial.traj"),
                    "write_debug_trajs": True,
                    "debug_traj_interval": 7,
                    "diagnostics_enabled": True,
                    "diagnostics_log": True,
                    "diagnostics_top_n": 2,
                }
            )
            cfg = SimpleNamespace(**cfg_dict)
            fake_pt = FakeReplicaExchange()

            workflow = AdsorbateReplicaExchangeWorkflow(
                cfg,
                snapshot_loader=lambda *_: _make_two_sided_mxene_like_slab(),
            )

            with mock.patch(
                "gcmc.workflows.ReplicaExchange.from_auto_config",
                return_value=fake_pt,
            ) as mocked_from_auto:
                pt = workflow.run()

            self.assertIs(pt, fake_pt)
            self.assertEqual(fake_pt.run_args, (3, 1))

            kwargs = mocked_from_auto.call_args.kwargs
            self.assertIs(kwargs["mc_class"], AdsorbateCMC)
            self.assertEqual(kwargs["mc_kwargs"]["adsorbate_anchor_index"], 0)
            self.assertEqual(kwargs["mc_kwargs"]["site_type"], ["atop"])
            self.assertEqual(kwargs["mc_kwargs"]["debug_traj_interval"], 7)
            self.assertTrue(kwargs["mc_kwargs"]["diagnostics_enabled"])
            self.assertTrue(kwargs["mc_kwargs"]["diagnostics_log"])
            self.assertEqual(kwargs["mc_kwargs"]["diagnostics_top_n"], 2)

            atoms_template = kwargs["atoms_template"]
            tags = np.asarray(atoms_template.get_tags(), dtype=int)
            self.assertTrue(np.any(tags >= ADSORBATE_TAG_OFFSET))

            initial_atoms = read(cfg.initial_traj_file)
            initial_tags = np.asarray(initial_atoms.get_tags(), dtype=int)
            self.assertTrue(np.any(initial_tags >= ADSORBATE_TAG_OFFSET))

            out_dir = Path(cfg.output_dir)
            self.assertEqual(
                fake_pt.replica_states[0]["traj_file"],
                str(out_dir / "replica_400K.traj"),
            )
            self.assertEqual(
                fake_pt.replica_states[1]["thermo_file"],
                str(out_dir / "replica_300K.dat"),
            )
            self.assertEqual(
                fake_pt.replica_states[1]["checkpoint_file"],
                str(out_dir / "checkpoint_300K.pkl"),
            )
            self.assertEqual(
                fake_pt.replica_states[0]["attempted_traj_file"],
                str(out_dir / "replica_400K_attempted.traj"),
            )
            self.assertEqual(
                fake_pt.replica_states[0]["accepted_traj_file"],
                str(out_dir / "replica_400K_accepted.traj"),
            )
            self.assertEqual(
                fake_pt.replica_states[0]["rejected_traj_file"],
                str(out_dir / "replica_400K_rejected.traj"),
            )

    def test_adsorbate_pt_workflow_passes_ray_backend_options(self):
        cfg = SimpleNamespace(
            **{
                **_DEFAULT_ADSORBATE_PT_CONFIG,
                "snapshot": "dummy.traj",
                "frame": 0,
                "adsorbate": "OH",
                "initialization_mode": "preloaded",
                "site_elements": ["Ti"],
                "site_type": ["atop"],
                "substrate_elements": ["Ti", "C"],
                "functional_elements": ["O"],
                "T_start": 500.0,
                "T_end": 300.0,
                "T_step": 100.0,
                "n_cycles": 2,
                "backend": "ray",
                "n_gpus": 2,
                "workers_per_gpu": 1,
                "ray_address": "auto",
                "ray_log_to_driver": False,
                "ray_num_cpus_per_task": 2,
                "ray_num_gpus_per_task": 0.5,
                "output_dir": "/tmp/ads_pt_out",
                "stats_file": "/tmp/ads_pt_out/stats.csv",
                "results_file": "/tmp/ads_pt_out/results.csv",
                "checkpoint_file": "/tmp/ads_pt_out/state.pkl",
                "initial_traj_file": "/tmp/ads_pt_out/initial.traj",
            }
        )
        fake_pt = FakeReplicaExchange()
        workflow = AdsorbateReplicaExchangeWorkflow(
            cfg,
            snapshot_loader=lambda *_: _make_two_sided_mxene_like_slab(),
        )

        with mock.patch(
            "gcmc.workflows.ReplicaExchange.from_auto_config",
            return_value=fake_pt,
        ) as mocked_from_auto:
            workflow.run()

        kwargs = mocked_from_auto.call_args.kwargs
        self.assertEqual(kwargs["execution_backend"], "ray")
        self.assertEqual(kwargs["n_gpus"], 2)
        self.assertEqual(kwargs["workers_per_gpu"], 1)
        self.assertEqual(
            kwargs["backend_kwargs"]["init_kwargs"],
            {"address": "auto", "log_to_driver": False},
        )
        self.assertEqual(
            kwargs["backend_kwargs"]["actor_options"],
            {"num_cpus": 2.0, "num_gpus": 0.5},
        )


if __name__ == "__main__":
    unittest.main()
