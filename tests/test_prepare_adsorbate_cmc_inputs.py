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
calculator:
  calculator: lj
""".strip()
            )
            defaults = _defaults_from_prepare_yaml(template)
            args = build_parser(defaults).parse_args([])

            _write_slurm(root, 1, args)
            script = (root / "run_array.slurm").read_text()

        self.assertEqual(defaults["signal_seconds_before_timeout"], 600)
        self.assertIn("#SBATCH --signal=B:USR1@600", script)
        self.assertIn("trap forward_stop_signal USR1", script)
        self.assertIn('kill -USR1 "${driver_pid}"', script)
        self.assertIn(
            'python3 -u -m gcmc.run_adsorbate_pt --config "${CONFIG}" &',
            script,
        )

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
