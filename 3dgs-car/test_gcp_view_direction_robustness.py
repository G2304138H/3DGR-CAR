import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gcp_view_direction_robustness import (
    condition_id,
    fit_quadratic_response_surface,
    resolve_robustness_plan,
    run_view_direction_robustness,
)


class RobustnessPlanTests(unittest.TestCase):
    def test_quadratic_response_surface_recovers_known_grid(self):
        samples = []
        for theta in (-2.0, 0.0, 2.0):
            for phi in (-3.0, 0.0, 3.0):
                value = 0.8 + 0.01 * theta - 0.02 * phi + 0.001 * theta * phi
                samples.append((theta, phi, value))
        fitted = fit_quadratic_response_surface(samples)
        self.assertIsNotNone(fitted)
        self.assertAlmostEqual(fitted["r_squared"], 1.0)
        self.assertAlmostEqual(
            fitted["coefficients"]["theta_phi_interaction"], 0.001
        )

    def test_default_grid_matches_parametric_signed_conditions(self):
        plan = resolve_robustness_plan({})
        self.assertEqual(len(plan["conditions"]), 24)
        self.assertIn((-15.0, 0.0), plan["conditions"])
        self.assertIn((10.0, -10.0), plan["conditions"])
        self.assertNotIn((0.0, 0.0), plan["conditions"])
        self.assertEqual(
            condition_id(-2.5, 10),
            "theta_change_minus2p5deg_phi_change_10deg",
        )

    def test_zero_condition_and_unplanned_visualization_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            resolve_robustness_plan(
                {"view_direction_robustness": {"changes_deg": [[0, 0]]}}
            )
        with self.assertRaisesRegex(ValueError, "must occur"):
            resolve_robustness_plan(
                {
                    "view_direction_robustness": {
                        "changes_deg": [[2, 0]],
                        "visualization_conditions_deg": [[5, 0]],
                    }
                }
            )

    def test_driver_compares_each_condition_to_accurate_k2_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_gcp.pt"
            checkpoint.write_bytes(b"checkpoint")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            baseline_cases = [
                {
                    "case_name": "lca_0001",
                    "split": "validation",
                    "eval_num_views": 2,
                }
            ]
            (baseline_dir / "performance_per_case.json").write_text(
                json.dumps(baseline_cases), encoding="utf-8"
            )
            baseline = {
                "comparison_condition": {
                    "checkpoint": str(checkpoint),
                    "evaluation_split": "val_test",
                    "effective_view_sweep": [
                        {"eval_num_views": 2, "view_indices": [0, 1]}
                    ],
                    "view_directions_accurate": True,
                    "evaluation_view_directions": {"accurate": True},
                },
                "roles_by_view_count": {
                    "k2": {
                        "optimized": {
                            "metrics": {
                                "masked_dice_3d": {
                                    "mean": 0.8,
                                    "standard_error": 0.01,
                                }
                            },
                            "timing": {"average_case_seconds": 10.0},
                        }
                    }
                },
                "per_case_metrics_file": "performance_per_case.json",
            }
            baseline_path = baseline_dir / "performance_summary.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            output = root / "robustness"
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            resolved = {
                "eval_output_dir": str(output),
                "checkpoint_path": str(checkpoint),
                "checkpoint_choice": "best",
                "experiment_dir": str(root),
                "eval_split": "val_test",
                "eval_num_views": 2,
                "eval_view_indices": [0, 1],
                "effective_view_sweep": [
                    {"eval_num_views": 2, "view_indices": [0, 1]}
                ],
                "evaluation_view_directions": {"accurate": True},
                "view_direction_robustness": {
                    "changes_deg": [[-2, 0], [2, 0]],
                    "visualization_conditions_deg": [[2, 0]],
                    "accurate_baseline_summary": str(baseline_path),
                    "require_accurate_baseline": True,
                },
            }

            def fake_run(command, cwd, check):
                child = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                child_output = Path(child["eval_output_dir"])
                child_output.mkdir(parents=True, exist_ok=True)
                theta = float(
                    child["evaluation_view_directions"]["theta_change_deg"]
                )
                cases = [dict(baseline_cases[0])]
                (child_output / "performance_per_case.json").write_text(
                    json.dumps(cases), encoding="utf-8"
                )
                performance = {
                    "comparison_condition": {
                        "checkpoint": str(checkpoint),
                        "evaluation_split": "val_test",
                        "effective_view_sweep": [
                            {"eval_num_views": 2, "view_indices": [0, 1]}
                        ],
                    },
                    "roles_by_view_count": {
                        "k2": {
                            "optimized": {
                                "metrics": {
                                    "masked_dice_3d": {
                                        "mean": 0.8 - abs(theta) / 100.0,
                                        "standard_error": 0.02,
                                    }
                                },
                                "timing": {"average_case_seconds": 11.0},
                            }
                        }
                    },
                    "per_case_metrics_file": "performance_per_case.json",
                }
                (child_output / "performance_summary.json").write_text(
                    json.dumps(performance), encoding="utf-8"
                )
                return mock.Mock(returncode=0)

            with mock.patch(
                "gcp_view_direction_robustness.subprocess.run",
                side_effect=fake_run,
            ):
                summary_path = run_view_direction_robustness(
                    resolved=resolved,
                    config_path=config_path,
                    evaluator_path=root / "evaluate_gcp.py",
                )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["num_conditions"], 2)
            self.assertEqual(summary["case_count"], 1)
            self.assertAlmostEqual(
                summary["results"][0]["comparison_to_accurate"]
                ["optimized"]["masked_dice_3d"]["signed_change"],
                -0.02,
            )
            self.assertTrue(
                (output / "view_direction_robustness_metrics.csv").is_file()
            )
            child_configs = sorted((output / "run_configs").glob("*.json"))
            self.assertEqual(len(child_configs), 2)
            child = json.loads(child_configs[0].read_text(encoding="utf-8"))
            self.assertEqual(child["eval_num_views"], 2)
            self.assertEqual(child["eval_view_indices"], [0, 1])
            self.assertFalse(child["evaluation_view_directions"]["accurate"])


if __name__ == "__main__":
    unittest.main()
