import tempfile
import unittest
from pathlib import Path

from gcmc.prepare_adsorbate_cmc_inputs import (
    _config_for_task,
    _defaults_from_prepare_yaml,
    _pt_config_for_task,
    build_parser,
)


class PrepareAdsorbateCMCInputsTests(unittest.TestCase):
    def test_debug_trajectory_settings_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.yaml"
            template.write_text(
                """
workflow: pt
cmc:
  write_debug_trajs: true
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

        self.assertTrue(defaults["write_debug_trajs"])
        self.assertEqual(defaults["debug_traj_interval"], 13)
        self.assertTrue(cmc_config["cmc"]["write_debug_trajs"])
        self.assertEqual(cmc_config["cmc"]["debug_traj_interval"], 13)
        self.assertTrue(pt_config["cmc"]["write_debug_trajs"])
        self.assertEqual(pt_config["cmc"]["debug_traj_interval"], 13)

    def test_cli_can_override_debug_trajectory_settings(self):
        args = build_parser(
            {"write_debug_trajs": True, "debug_traj_interval": 13}
        ).parse_args(["--no-write-debug-trajs", "--debug-traj-interval", "5"])

        self.assertFalse(args.write_debug_trajs)
        self.assertEqual(args.debug_traj_interval, 5)


if __name__ == "__main__":
    unittest.main()
