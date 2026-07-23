from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.active_learning.evaluation import build_parser as evaluation_parser
from tools.active_learning.models import discover_model_paths, resolve_models
from tools.active_learning.selection import (
    build_parser as selection_parser,
    resolve_reference_pools,
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

    def test_evaluation_uses_same_model_path_interface(self):
        args = evaluation_parser().parse_args(["--model-path", "committee"])

        self.assertEqual(args.model_path, ["committee"])
        self.assertEqual(args.calculator, "auto")


if __name__ == "__main__":
    unittest.main()
