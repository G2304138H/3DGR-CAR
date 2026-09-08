import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from evaluate_gcp import (
    build_stage2_arguments,
    main,
    parse_args,
    resolve_evaluation_config,
    resolve_view_sweep,
)
from evaluate_stage2_npz import METRIC_NAMES, parse_args as parse_stage2_args


class GcpEvaluationConfigTests(unittest.TestCase):
    def test_concrete_lca_and_rca_configs_are_self_contained_val_test_jobs(self):
        config_dir = Path(__file__).resolve().parent / "configs"
        expected = {
            "lca": {"spacing": 0.65, "sid": None},
            "rca": {"spacing": 0.55, "sid": 0.9},
        }
        for artery, calibration in expected.items():
            path = (
                config_dir
                / f"eval_gcp_paper_metric_{artery}_val_test.json"
            )
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["evaluation_mode"], "paper_metric")
            self.assertEqual(config["eval_split"], "val_test")
            model = config["model"]
            self.assertEqual(model["checkpoint_choice"], "best")
            self.assertTrue(model["pretrained_weights"].endswith("/best_gcp.pt"))
            self.assertEqual(
                model["parameters"],
                {
                    "image_size": 128,
                    "in_channels": 1,
                    "base_channels": 32,
                    "num_levels": 4,
                    "alpha": 2,
                    "offset_scale": 0.1,
                    "norm_groups": 8,
                    "dropout": 0.0,
                },
            )
            self.assertEqual(config["eval_num_views"], [1, 2, 4])
            self.assertEqual(config["eval_view_indices"], [0, 1, 2, 3])
            self.assertTrue(config["continue_on_error"])
            optimization = config["gaussian_optimization"]
            self.assertEqual(
                optimization["fallback_detector_pixel_spacing_mm"],
                calibration["spacing"],
            )
            self.assertEqual(
                optimization.get("fallback_sid_m"), calibration["sid"]
            )

    def _fixture(self, root: Path, mode: str = "paper_metric") -> Path:
        experiment = root / "experiment"
        projections = root / "projections"
        ground_truth = root / "ground_truth"
        output = root / "evaluation"
        experiment.mkdir()
        projections.mkdir()
        ground_truth.mkdir()
        (experiment / "best_gcp.pt").write_bytes(b"checkpoint")
        split = root / "split.json"
        split.write_text(
            json.dumps({"val": ["lca_0001"], "test": ["lca_0002"]}),
            encoding="utf-8",
        )
        (experiment / "training_config.json").write_text(
            json.dumps(
                {
                    "projection_dir": str(projections),
                    "ground_truth_dir": str(ground_truth),
                    "split_json": str(split),
                    "fallback_detector_pixel_spacing_mm": 0.65,
                    "fallback_sid_m": 0.9,
                }
            ),
            encoding="utf-8",
        )
        config = root / "eval.json"
        config.write_text(
            json.dumps(
                {
                    "model": {
                        "experiment_dir": str(experiment),
                        "pretrained_weights": str(experiment / "best_gcp.pt"),
                        "checkpoint_choice": "best",
                        "parameters": {
                            "image_size": 128,
                            "in_channels": 1,
                            "base_channels": 32,
                            "num_levels": 4,
                            "alpha": 2,
                            "offset_scale": 0.1,
                            "norm_groups": 8,
                            "dropout": 0.0,
                        },
                        "parameterization": {
                            "depth_activation": "sigmoid",
                            "offset_activation": "bounded_tanh",
                            "coordinate_order": "normalized_zyx",
                            "initialization_view": "first_selected_view",
                        },
                    },
                    "eval_output_dir": str(output),
                    "evaluation_mode": mode,
                    "eval_split": "val_test",
                    "eval_num_views": [1, 2],
                    "eval_view_indices": [3, 5],
                    "paper_metric_ground_truth_dir": None,
                    "paper_metric_volume_threshold": None,
                    "paper_metric_prediction_threshold_percentile": 97,
                    "gaussian_optimization": {
                        "iterations": 12,
                        "early_stop_checks": 3,
                        "densify": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        return config

    def test_view_count_sweep_uses_fixed_prefixes(self):
        self.assertEqual(
            resolve_view_sweep(
                {
                    "eval_num_views": [1, 2, 4],
                    "eval_view_indices": [6, 4, 2, 0],
                }
            ),
            [(1, [6]), (2, [6, 4]), (4, [6, 4, 2, 0])],
        )

    def test_resolves_checkpoint_and_training_dataset_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._fixture(Path(directory))
            resolved = resolve_evaluation_config(config_path)
            experiment = Path(directory) / "experiment"
            self.assertEqual(
                resolved["checkpoint_path"],
                str((experiment / "best_gcp.pt").resolve()),
            )
            self.assertEqual(
                resolved["evaluation_dataset_dir"],
                str((Path(directory) / "projections").resolve()),
            )
            self.assertEqual(
                resolved["paper_metric_ground_truth_dir"],
                str((Path(directory) / "ground_truth").resolve()),
            )
            self.assertEqual(
                resolved["effective_view_sweep"],
                [
                    {"eval_num_views": 1, "view_indices": [3]},
                    {"eval_num_views": 2, "view_indices": [3, 5]},
                ],
            )

    def test_cli_case_and_split_overrides_replace_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._fixture(Path(directory))
            resolved = resolve_evaluation_config(
                config_path,
                eval_case_ids_override=["lca_0042"],
                eval_split_override="test",
            )
            self.assertEqual(resolved["eval_case_ids"], ["lca_0042"])
            self.assertEqual(resolved["eval_split"], "test")

    def test_four_view_cli_override_builds_only_k4(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._fixture(Path(directory))
            resolved = resolve_evaluation_config(
                config_path,
                eval_num_views_override=[4],
                eval_view_indices_override=[0, 1, 2, 3],
            )
            self.assertEqual(resolved["eval_num_views"], [4])
            self.assertEqual(resolved["eval_view_indices"], [0, 1, 2, 3])
            self.assertEqual(
                resolved["effective_view_sweep"],
                [
                    {
                        "eval_num_views": 4,
                        "view_indices": [0, 1, 2, 3],
                    }
                ],
            )

    def test_case_number_cli_alias_is_repeatable(self):
        args = parse_args(
            [
                "--config",
                "evaluation.json",
                "--case-number",
                "42",
                "--case-id",
                "lca_0043",
                "--split",
                "val",
                "--num-views",
                "4",
                "--view-indices",
                "0",
                "1",
                "2",
                "3",
            ]
        )
        self.assertEqual(args.case_ids, ["42", "lca_0043"])
        self.assertEqual(args.split, "val")
        self.assertEqual(args.eval_num_views, [4])
        self.assertEqual(args.view_indices, [0, 1, 2, 3])

    def test_stage2_arguments_force_gcp_and_keep_all_optimization_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = resolve_evaluation_config(self._fixture(root))
            arguments = build_stage2_arguments(
                resolved,
                run_output_dir=root / "run",
                view_indices=[3, 5],
            )
            view_position = arguments.index("--view-indices")
            self.assertEqual(arguments[view_position + 1 : view_position + 3], ["3", "5"])
            init_position = arguments.index("--init-method")
            self.assertEqual(arguments[init_position + 1], "gcp")
            checkpoint_position = arguments.index("--gcp-checkpoint")
            self.assertEqual(
                arguments[checkpoint_position + 1], resolved["checkpoint_path"]
            )
            expected_position = arguments.index(
                "--expected-gcp-model-config-json"
            )
            self.assertEqual(
                json.loads(arguments[expected_position + 1]),
                resolved["model"]["parameters"],
            )
            self.assertIn("--save-evaluation-arrays", arguments)
            self.assertIn("--no-densify", arguments)
            self.assertEqual(arguments[arguments.index("--iterations") + 1], "12")
            self.assertEqual(
                arguments[
                    arguments.index("--fallback-detector-pixel-spacing-mm") + 1
                ],
                "0.65",
            )
            self.assertEqual(
                arguments[arguments.index("--fallback-sid-m") + 1], "0.9"
            )
            stage2_args, trainer_args = parse_stage2_args(arguments)
            self.assertEqual(stage2_args.view_indices, [3, 5])
            self.assertIn("--gcp-checkpoint", trainer_args)

    def test_all_unselected_novel_views_preserves_trainer_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = resolve_evaluation_config(
                self._fixture(root, mode="visualisation")
            )
            resolved["gaussian_optimization"]["novel_view_indices"] = (
                "all_unselected"
            )
            arguments = build_stage2_arguments(
                resolved,
                run_output_dir=root / "run",
                view_indices=[3, 5],
            )
            self.assertNotIn("--novel-view-indices", arguments)

    def test_dry_run_writes_resolved_config_and_plan_without_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._fixture(root, mode="visualization")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(config_path), "--dry-run"]), 0)
            output = root / "evaluation"
            self.assertTrue((output / "resolved_config.json").is_file())
            plan = json.loads(
                (output / "evaluation_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["evaluation_mode"], "visualisation")
            self.assertEqual([run["view_label"] for run in plan["runs"]], ["k1", "k2"])
            self.assertTrue(
                all("--init-method" in run["stage2_arguments"] for run in plan["runs"])
            )

    def test_paper_mode_combines_each_view_count_into_parametric_style_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._fixture(root)

            def fake_stage2(stage2_arguments):
                run_dir = Path(
                    stage2_arguments[stage2_arguments.index("--output-dir") + 1]
                )
                view_values = []
                cursor = stage2_arguments.index("--view-indices") + 1
                while cursor < len(stage2_arguments) and not stage2_arguments[cursor].startswith("--"):
                    view_values.append(int(stage2_arguments[cursor]))
                    cursor += 1
                count = len(view_values)
                run_dir.mkdir(parents=True, exist_ok=True)
                case = {
                    "case_name": "lca_0001",
                    "split": "val_test",
                    "status": "completed",
                    "case_wall_time_seconds": float(count * 10),
                    "reconstruction_wall_time_seconds": float(count * 8),
                    "optimization_elapsed_seconds": float(count * 6),
                    "metrics_wall_time_seconds": float(count * 2),
                    **{name: float(count) for name in METRIC_NAMES},
                }
                summary = {
                    "num_cases_requested": 1,
                    "num_cases_completed": 1,
                    "num_cases_failed": 0,
                    "metrics": {
                        name: {"mean": float(count), "standard_error": None}
                        for name in METRIC_NAMES
                    },
                }
                (run_dir / "evaluation_results.json").write_text(
                    json.dumps(
                        {
                            "format": "3dgr_car_stage2_evaluation_v2",
                            "summary": summary,
                            "matrix": {},
                            "cases": [case],
                        }
                    ),
                    encoding="utf-8",
                )
                return 0

            with mock.patch(
                "evaluate_gcp.stage2_evaluation.main", side_effect=fake_stage2
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--config", str(config_path)]), 0)

            output = root / "evaluation"
            performance = json.loads(
                (output / "performance_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(performance["evaluation"]["num_cases"], 1)
            self.assertEqual(
                performance["evaluation"]["num_case_view_evaluations"], 2
            )
            self.assertEqual(
                sorted(performance["evaluation"]["metrics_by_view_count"]),
                ["k1", "k2"],
            )
            timing = performance["evaluation"]["timing"]
            self.assertEqual(timing["average_case_seconds"], 15.0)
            self.assertEqual(
                timing["average_pipeline_seconds_per_case"], 12.0
            )
            self.assertEqual(
                timing["average_optimization_seconds_per_case"], 9.0
            )
            self.assertEqual(
                timing["by_view_count"]["k1"]["average_case_seconds"],
                10.0,
            )
            self.assertEqual(
                timing["by_view_count"]["k2"]["average_case_seconds"],
                20.0,
            )
            self.assertTrue(
                (output / "metrics" / "paper_metric_per_case.json").is_file()
            )
            paper_summary = json.loads(
                (output / "metrics" / "paper_metric_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(paper_summary["timing"], timing)
            self.assertTrue((output / "metrics" / "metrics_matrix.npz").is_file())


if __name__ == "__main__":
    unittest.main()
