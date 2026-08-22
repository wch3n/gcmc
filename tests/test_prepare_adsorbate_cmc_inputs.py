import tempfile
import unittest
from pathlib import Path

from gcmc.prepare_adsorbate_cmc_inputs import (
    _config_for_task,
    _defaults_from_prepare_yaml,
    _pt_config_for_task,
    _write_slurm,
    build_parser,
)


class PrepareAdsorbateCMCInputsTests(unittest.TestCase):
    def test_enabled_puckering_block_does_not_require_legacy_probability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
moves:
  hop:
    prob: 0.8
    puckering:
      enabled: true
      elements: [Ti, Zr]
      height_A: 1.0
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])
            config = _pt_config_for_task(
                snapshot_path=root / "snapshots" / "snapshot.traj",
                run_dir=root / "runs" / "snapshot_000" / "seed_067",
                source_snapshot=root / "source.traj",
                source_frame=4,
                source_cycle=20,
                seed=67,
                args=args,
            )

        self.assertEqual(defaults["puckering_prob"], 1.0)
        puckering = config["cmc"]["moves"]["hop"]["puckering"]
        self.assertEqual(puckering["prob"], 1.0)
        self.assertEqual(puckering["elements"], ["Ti", "Zr"])

    def test_element_specific_puckering_heights_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
moves:
  hop:
    prob: 0.8
    puckering:
      prob: 0.25
      elements: [Ti, Zr]
      height_A: 0.5
      height_jitter_A: 0.1
      heights:
        Ti:
          height_A: 1.2
          height_jitter_A: 0.3
        Zr:
          height_A: 0.55
          height_jitter_A: 0.15
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])
            config = _pt_config_for_task(
                snapshot_path=root / "snapshots" / "snapshot.traj",
                run_dir=root / "runs" / "snapshot_000" / "seed_067",
                source_snapshot=root / "source.traj",
                source_frame=4,
                source_cycle=20,
                seed=67,
                args=args,
            )

        expected = {
            "Ti": {"height_A": 1.2, "height_jitter_A": 0.3},
            "Zr": {"height_A": 0.55, "height_jitter_A": 0.15},
        }
        self.assertEqual(defaults["puckering_heights"], expected)
        self.assertEqual(
            config["cmc"]["moves"]["hop"]["puckering"]["heights"],
            expected,
        )

    def test_pt_slurm_forwards_pre_timeout_signal_to_driver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
backend:
  backend: ray
slurm:
  signal_seconds_before_timeout: 600
  auto_requeue: true
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])

            _write_slurm(root, 1, args)
            script = (root / "run_array.slurm").read_text()

        self.assertEqual(defaults["signal_seconds_before_timeout"], 600)
        self.assertTrue(defaults["auto_requeue"])
        self.assertIn("#SBATCH --signal=B:USR1@600", script)
        self.assertIn("#SBATCH --requeue", script)
        self.assertIn("#SBATCH --open-mode=append", script)
        self.assertIn("trap forward_stop_signal USR1", script)
        self.assertIn('kill -USR1 "${driver_pid}"', script)
        self.assertIn('scontrol requeue "${requeue_target}"', script)
        self.assertIn("refusing automatic requeue", script)
        self.assertIn(
            'python3 -u -m gcmc.run_adsorbate_pt --config "${CONFIG}" &',
            script,
        )

    def test_auto_requeue_requires_pre_timeout_signal(self):
        args = build_parser(
            {
                "workflow": "pt",
                "auto_requeue": True,
                "signal_seconds_before_timeout": 0,
            }
        ).parse_args([])

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError,
                "signal_seconds_before_timeout > 0",
            ):
                _write_slurm(Path(tmpdir), 1, args)

    def test_resume_settings_are_propagated_to_generated_workflows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
cmc:
  resume: true
pt:
  resume: true
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])
            common = {
                "snapshot_path": root / "snapshots" / "snapshot.traj",
                "run_dir": root / "runs" / "snapshot_000" / "seed_067",
                "source_snapshot": root / "source.traj",
                "source_frame": 4,
                "source_cycle": 20,
                "seed": 67,
                "args": args,
            }

            cmc_config = _config_for_task(**common)
            pt_config = _pt_config_for_task(**common)

        self.assertTrue(defaults["cmc_resume"])
        self.assertTrue(cmc_config["cmc"]["resume"])
        self.assertTrue(pt_config["pt"]["resume"])
        self.assertNotIn("resume", pt_config["cmc"])

    def test_debug_trajectory_settings_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
cmc:
  write_debug_trajs: false
  write_attempted_traj: false
  write_accepted_traj: true
  write_rejected_traj: true
  debug_traj_interval: 13
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])

            common = {
                "snapshot_path": root / "snapshots" / "snapshot.traj",
                "run_dir": root / "runs" / "snapshot_000" / "seed_067",
                "source_snapshot": root / "source.traj",
                "source_frame": 4,
                "source_cycle": 20,
                "seed": 67,
                "args": args,
            }
            cmc_config = _config_for_task(**common)
            pt_config = _pt_config_for_task(**common)

        self.assertFalse(defaults["write_debug_trajs"])
        self.assertFalse(defaults["write_attempted_traj"])
        self.assertTrue(defaults["write_accepted_traj"])
        self.assertTrue(defaults["write_rejected_traj"])
        self.assertEqual(defaults["debug_traj_interval"], 13)
        self.assertFalse(cmc_config["cmc"]["write_debug_trajs"])
        self.assertFalse(cmc_config["cmc"]["write_attempted_traj"])
        self.assertTrue(cmc_config["cmc"]["write_accepted_traj"])
        self.assertTrue(cmc_config["cmc"]["write_rejected_traj"])
        self.assertEqual(cmc_config["cmc"]["debug_traj_interval"], 13)
        self.assertFalse(pt_config["cmc"]["write_debug_trajs"])
        self.assertFalse(pt_config["cmc"]["write_attempted_traj"])
        self.assertTrue(pt_config["cmc"]["write_accepted_traj"])
        self.assertTrue(pt_config["cmc"]["write_rejected_traj"])
        self.assertEqual(pt_config["cmc"]["debug_traj_interval"], 13)

    def test_cli_can_override_debug_trajectory_settings(self):
        args = build_parser(
            {
                "write_debug_trajs": True,
                "write_accepted_traj": False,
                "debug_traj_interval": 13,
            }
        ).parse_args(
            [
                "--no-write-debug-trajs",
                "--write-accepted-traj",
                "--debug-traj-interval",
                "5",
            ]
        )

        self.assertFalse(args.write_debug_trajs)
        self.assertTrue(args.write_accepted_traj)
        self.assertEqual(args.debug_traj_interval, 5)


if __name__ == "__main__":
    unittest.main()
