import json
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gcp_translation_calibration_robustness import (
    resolve_translation_plan,
    run_translation_calibration_robustness,
    translation_condition_id,
)
from stage2_translation_projection import (
    _render_view,
    create_translated_stage2_case,
    translation_vector_mm,
)


class TranslationCalibrationTests(unittest.TestCase):
    def test_fixed_plan_has_the_nine_positive_norm_preserving_conditions(self):
        plan = resolve_translation_plan({})
        self.assertEqual(len(plan["conditions"]), 9)
        self.assertEqual(
            [item["family"] for item in plan["conditions"]],
            ["Y", "Y", "Y", "XZ", "XZ", "XZ", "XYZ", "XYZ", "XYZ"],
        )
        for condition in plan["conditions"]:
            self.assertAlmostEqual(
                np.linalg.norm(condition["vector_xyz_mm"]),
                condition["magnitude_mm"],
            )
            self.assertTrue(all(value >= 0.0 for value in condition["vector_xyz_mm"]))
        self.assertEqual(translation_condition_id("XYZ", 20), "xyz_20mm")

    def test_translation_formulas_use_total_euclidean_magnitude(self):
        self.assertEqual(translation_vector_mm("Y", 5), (0.0, 5.0, 0.0))
        self.assertEqual(
            translation_vector_mm("XZ", 10),
            (10 / math.sqrt(2), 0.0, 10 / math.sqrt(2)),
        )
        self.assertEqual(
            translation_vector_mm("XYZ", 20),
            (20 / math.sqrt(3),) * 3,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("skimage") is not None,
        "scikit-image is not installed in the lightweight test environment",
    )
    def test_only_second_selected_image_is_replaced_and_angles_stay_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.linspace(-0.025, 0.025, 20, dtype=np.float32)
            artery = np.zeros((1, 20, 4), dtype=np.float32)
            artery[0, :, 0] = points
            artery[0, :, 1] = 0.004
            artery[0, :, 2] = 0.006
            artery[0, :, 3] = 0.0015
            payload = {
                "sample_name": np.asarray("rca_0001"),
                "artery": artery,
                "images": np.zeros((2, 64, 64), dtype=np.float32),
                "theta_deg": np.asarray([0.0, 40.0], dtype=np.float32),
                "phi_deg": np.asarray([0.0, 10.0], dtype=np.float32),
                "view_features": np.asarray(
                    [[0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
                    dtype=np.float32,
                ),
                "projection_center_offset": np.zeros(3, dtype=np.float32),
                "projected_branch_indices": np.asarray([0], dtype=np.int32),
                "sid": np.asarray(0.9, dtype=np.float32),
                "imager_pixel_spacing": np.asarray(0.65, dtype=np.float32),
                "imager_pixel_spacing_units": np.asarray("mm"),
                "mask_render_mode": np.asarray("filled"),
            }
            first, _ = _render_view(
                payload,
                view_index=0,
                translation_xyz_m=np.zeros(3),
                num_circle_points=120,
            )
            second, _ = _render_view(
                payload,
                view_index=1,
                translation_xyz_m=np.zeros(3),
                num_circle_points=120,
            )
            payload["images"] = np.stack((first, second), axis=0)
            source = root / "source.npz"
            output = root / "translated.npz"
            np.savez_compressed(source, **payload)

            metadata = create_translated_stage2_case(
                source,
                output,
                selected_view_indices=[0, 1],
                translation_xyz_mm=[0.0, 10.0, 0.0],
            )
            self.assertAlmostEqual(metadata["translation_clean_rerender_dice"], 1.0)
            self.assertEqual(metadata["translation_delta_theta_deg"], 0.0)
            self.assertEqual(metadata["translation_delta_phi_deg"], 0.0)
            self.assertFalse(metadata["translation_image_feature_cache_used"])

            with np.load(output, allow_pickle=False) as translated:
                self.assertTrue(np.array_equal(translated["images"][0], first))
                self.assertFalse(np.array_equal(translated["images"][1], second))
                self.assertTrue(
                    np.array_equal(translated["theta_deg"], payload["theta_deg"])
                )
                self.assertTrue(
                    np.array_equal(translated["phi_deg"], payload["phi_deg"])
                )
                self.assertTrue(
                    np.array_equal(
                        translated["view_features"], payload["view_features"]
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        translated["projection_center_offset"],
                        payload["projection_center_offset"],
                    )
                )

    def test_concrete_configs_are_fixed_two_view_jobs(self):
        config_dir = Path(__file__).resolve().parent / "configs"
        for artery in ("lca", "rca"):
            config = json.loads(
                (
                    config_dir
                    / f"eval_gcp_translational_calibration_{artery}_val_test.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                config["evaluation_mode"],
                "translational_calibration_robustness",
            )
            self.assertEqual(config["eval_num_views"], 2)
            self.assertEqual(config["eval_view_indices"], [0, 1])
            self.assertEqual(
                config["evaluation_view_directions"],
                {
                    "accurate": True,
                    "theta_change_deg": 0.0,
                    "phi_change_deg": 0.0,
                },
            )
            robustness = config["translational_calibration_robustness"]
            self.assertEqual(robustness["clean_rerender_min_dice"], 0.98)
            self.assertEqual(robustness["renderer_num_circle_points"], 120)

    def test_driver_aggregates_all_conditions_and_gcp_optimization_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_gcp.pt"
            checkpoint.write_bytes(b"checkpoint")
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            metric_names = (
                "masked_dice_3d",
                "mse_3d",
                "ssim_3d",
                "masked_mse",
                "masked_mae",
                "masked_psnr",
                "masked_ssim_3d",
            )

            def case_record(vector=None, optimized=0.8):
                record = {
                    "case_name": "lca_0001",
                    "split": "validation",
                    "status": "completed",
                    "eval_num_views": 2,
                }
                for name in metric_names:
                    record[name] = optimized
                    record[f"gcp_initial_{name}"] = optimized - 0.1
                    record[f"optimized_minus_gcp_initial_{name}"] = 0.1
                if vector is not None:
                    record.update(
                        {
                            "translation_selected_view_indices": [0, 1],
                            "translation_vector_xyz_mm": vector,
                            "translation_delta_theta_deg": 0.0,
                            "translation_delta_phi_deg": 0.0,
                            "translation_clean_rerender_dice": 0.995,
                            "translation_visible_centerline_fraction": 0.9,
                            "translation_visible_vessel_surface_fraction": 0.88,
                            "translation_original_view2_foreground_pixel_ratio": 0.1,
                            "translation_zero_rerender_view2_foreground_pixel_ratio": 0.1,
                            "translation_translated_view2_foreground_pixel_ratio": 0.08,
                            "translation_zero_visible_centerline_fraction": 1.0,
                            "translation_zero_visible_vessel_surface_fraction": 1.0,
                        }
                    )
                return record

            def role_summary(value):
                return {
                    "metrics": {
                        name: {"mean": value, "standard_error": 0.0}
                        for name in metric_names
                    }
                }

            baseline_case = case_record()
            (baseline_dir / "performance_per_case.json").write_text(
                json.dumps([baseline_case]), encoding="utf-8"
            )
            baseline = {
                "comparison_condition": {
                    "checkpoint": str(checkpoint),
                    "evaluation_split": "val_test",
                    "effective_view_sweep": [
                        {"eval_num_views": 2, "view_indices": [0, 1]}
                    ],
                    "evaluation_view_directions": {"accurate": True},
                    "view2_translation_mm": None,
                },
                "roles_by_view_count": {
                    "k2": {
                        "optimized": role_summary(0.8),
                        "gcp_initialization": role_summary(0.7),
                        "optimized_minus_gcp_initialization": role_summary(0.1),
                    }
                },
                "per_case_metrics_file": "performance_per_case.json",
            }
            baseline_path = baseline_dir / "performance_summary.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            output = root / "translation"
            resolved = {
                "eval_output_dir": str(output),
                "checkpoint_path": str(checkpoint),
                "checkpoint_choice": "best",
                "experiment_dir": str(root),
                "eval_split": "val_test",
                "effective_view_sweep": [
                    {"eval_num_views": 2, "view_indices": [0, 1]}
                ],
                "evaluation_view_directions": {
                    "accurate": True,
                    "theta_change_deg": 0.0,
                    "phi_change_deg": 0.0,
                },
                "reuse_existing": False,
                "translational_calibration_robustness": {
                    "accurate_baseline_summary": str(baseline_path),
                },
            }
            config_path = root / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            def fake_run(command, cwd, check):
                child = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                child_output = Path(child["eval_output_dir"])
                child_output.mkdir(parents=True, exist_ok=True)
                vector = child["view2_translation_mm"]
                current_case = case_record(vector=vector, optimized=0.75)
                (child_output / "performance_per_case.json").write_text(
                    json.dumps([current_case]), encoding="utf-8"
                )
                performance = {
                    "comparison_condition": {
                        "checkpoint": str(checkpoint),
                        "evaluation_split": "val_test",
                        "effective_view_sweep": [
                            {"eval_num_views": 2, "view_indices": [0, 1]}
                        ],
                        "evaluation_view_directions": {
                            "accurate": True,
                            "theta_change_deg": 0.0,
                            "phi_change_deg": 0.0,
                        },
                        "view2_translation_mm": vector,
                    },
                    "roles_by_view_count": {
                        "k2": {
                            "optimized": role_summary(0.75),
                            "gcp_initialization": role_summary(0.65),
                            "optimized_minus_gcp_initialization": role_summary(0.1),
                        }
                    },
                    "per_case_metrics_file": "performance_per_case.json",
                }
                (child_output / "performance_summary.json").write_text(
                    json.dumps(performance), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0)

            with mock.patch(
                "gcp_translation_calibration_robustness.subprocess.run",
                side_effect=fake_run,
            ):
                summary_path = run_translation_calibration_robustness(
                    resolved=resolved,
                    config_path=config_path,
                    evaluator_path=root / "evaluate_gcp.py",
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["num_conditions"], 9)
            self.assertIn("gcp_initialization", summary["results"][0]["roles"])
            self.assertAlmostEqual(
                summary["results"][0]["comparison_to_accurate"]["optimized"]
                ["masked_dice_3d"]["signed_change"],
                -0.05,
            )
            self.assertTrue(
                (output / "translation_calibration_robustness_metrics.csv").is_file()
            )
            self.assertTrue(
                (output / "translation_calibration_robustness_per_case.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
