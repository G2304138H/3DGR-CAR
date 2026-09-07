import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from visualize_gcp_case import artery_config_path, main


class SingleCaseVisualizationTests(unittest.TestCase):
    def test_artery_configs_are_full_single_case_visualization_jobs(self):
        expected_calibration = {
            "lca": {"spacing": 0.65, "sid": None},
            "rca": {"spacing": 0.55, "sid": 0.9},
        }
        for artery, calibration in expected_calibration.items():
            path = artery_config_path(artery)
            self.assertTrue(path.is_file())
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["evaluation_mode"], "visualisation")
            self.assertEqual(config["eval_case_ids"], ["1"])
            self.assertEqual(config["max_visualizations"], 1)
            self.assertTrue(
                config["model"]["pretrained_weights"].endswith("/best_gcp.pt")
            )
            optimization = config["gaussian_optimization"]
            self.assertEqual(
                optimization["fallback_detector_pixel_spacing_mm"],
                calibration["spacing"],
            )
            self.assertEqual(optimization["fallback_sid_m"], calibration["sid"])
            self.assertEqual(
                optimization["novel_view_indices"], "all_unselected"
            )
            self.assertTrue(optimization["save_volume_gif"])

    def test_case_number_selects_config_and_case_specific_output(self):
        with mock.patch(
            "visualize_gcp_case.evaluate_gcp.main", return_value=0
        ) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--artery",
                        "rca",
                        "--case-number",
                        "508",
                        "--dry-run",
                    ]
                )

        self.assertEqual(result, 0)
        arguments = evaluate.call_args.args[0]
        self.assertEqual(
            arguments[arguments.index("--case-id") + 1], "rca_0508"
        )
        config_path = Path(arguments[arguments.index("--config") + 1])
        self.assertEqual(
            config_path.name, "eval_gcp_visualisation_rca_case.json"
        )
        output_path = Path(arguments[arguments.index("--output-dir") + 1])
        self.assertEqual(output_path.name, "rca_0508")
        self.assertIn("--dry-run", arguments)


if __name__ == "__main__":
    unittest.main()
