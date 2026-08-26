import json
import contextlib
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from evaluate_stage2_npz import (
    NpzIndex,
    compute_volume_metrics,
    ground_truth_case_references,
    load_ground_truth_volume,
    load_split_case_references,
    main,
    parse_args,
    positive_percentile_threshold,
    projection_offset_to_voxel_shift_zyx,
    resample_ground_truth_to_prediction_grid,
    structural_similarity_3d,
    translate_volume_zyx,
)


class SplitLoadingTests(unittest.TestCase):
    def test_json_only_run_removes_case_cache_and_embeds_timing_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projections = root / "projections"
            ground_truths = root / "ground_truths"
            output = root / "output"
            projections.mkdir()
            (ground_truths / "rca").mkdir(parents=True)
            volume = np.zeros((9, 9, 9), dtype=np.float32)
            volume[3:6, 3:6, 3:6] = 1.0
            np.savez(
                projections / "rca_0001.npz",
                sample_name="rca_0001",
                vessel_type="rca",
                case_id="1",
            )
            np.savez(ground_truths / "rca" / "1.npz", voxel=volume)
            split = root / "split.json"
            split.write_text(json.dumps({"test": ["rca_0001"]}), encoding="utf-8")

            def fake_reconstruction(_script, _projection, case_output, _args):
                np.save(case_output / "reconstructed_volume_zyx.npy", volume)
                (case_output / "optimization_timing.json").write_text(
                    json.dumps(
                        {
                            "elapsed_seconds": 12.5,
                            "seconds_per_iteration": 0.025,
                            "iterations_requested": 8000,
                            "iterations_completed": 500,
                            "early_stopped": True,
                            "gpu_name": "test-gpu",
                        }
                    ),
                    encoding="utf-8",
                )

            argv = [
                "--input-dir", str(projections),
                "--split-json", str(split),
                "--split", "test",
                "--output-dir", str(output),
                "--ground-truth-dir", str(ground_truths),
            ]
            with mock.patch(
                "evaluate_stage2_npz.run_reconstruction",
                side_effect=fake_reconstruction,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(argv), 0)

            files = sorted(
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            )
            self.assertEqual(files, ["evaluation_results.json"])
            results = json.loads(
                (output / "evaluation_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(results["matrix"]["case_names"], ["rca_0001"])
            self.assertEqual(len(results["matrix"]["values"]), 1)
            self.assertEqual(
                results["cases"][0]["optimization_elapsed_seconds"],
                12.5,
            )
            self.assertTrue(results["cases"][0]["case_cache_removed"])
            self.assertGreater(
                results["summary"]["timing"]["average_case_seconds"],
                0.0,
            )

    def test_split_evaluation_requires_positive_early_stopping_patience(self):
        required = [
            "--input-dir", "input",
            "--split-json", "split.json",
            "--split", "test",
            "--output-dir", "output",
            "--ground-truth-dir", "gt",
        ]
        args, _ = parse_args(required)
        self.assertEqual(args.early_stop_checks, 7)
        self.assertIsNone(args.prediction_threshold)
        self.assertEqual(args.prediction_threshold_percentile, 97.0)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([*required, "--early-stop-checks", "0"])

    def test_nested_validation_alias_and_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps({"dataset": {"splits": {"val": [{"case_number": 1}, {"name": "rca_0002"}]}}}),
                encoding="utf-8",
            )
            self.assertEqual(load_split_case_references(path, "validation"), ["1", "rca_0002"])

    def test_case_number_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "rca_0007.npz", volume=np.zeros((3, 3, 3)))
            self.assertEqual(NpzIndex(root).resolve(["7"], "projection").name, "rca_0007.npz")

    def test_vessel_subdirectory_disambiguates_numeric_gt_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection = root / "lca_0001.npz"
            np.savez(
                projection,
                sample_name="lca_0001",
                vessel_type="lca",
                case_id="1",
            )
            ground_truth_root = root / "ground_truth"
            (ground_truth_root / "lca").mkdir(parents=True)
            (ground_truth_root / "rca").mkdir(parents=True)
            np.savez(ground_truth_root / "lca" / "1.npz", vol=np.zeros((3, 3, 3)))
            np.savez(ground_truth_root / "rca" / "1.npz", vol=np.zeros((3, 3, 3)))
            references, vessel_type, case_id = ground_truth_case_references(
                projection,
                "lca_0001",
                "lca_0001.npz",
            )
            resolved = NpzIndex(ground_truth_root).resolve(references, "ground-truth")
            self.assertEqual(vessel_type, "lca")
            self.assertEqual(case_id, "1")
            self.assertEqual(resolved, (ground_truth_root / "lca" / "1.npz").resolve())

    def test_split_key_with_case_numbers_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"test_case_numbers": [3, 8]}), encoding="utf-8")
            self.assertEqual(load_split_case_references(path, "test"), ["3", "8"])


class MetricTests(unittest.TestCase):
    def test_p97_prediction_mask_matches_gif_positive_percentile_rule(self):
        prediction = np.zeros((10, 10, 10), dtype=np.float32)
        prediction.reshape(-1)[:100] = np.arange(1, 101, dtype=np.float32)
        threshold = positive_percentile_threshold(prediction, 97.0)
        ground_truth = (prediction > threshold).astype(np.float32)

        metrics, arrays = compute_volume_metrics(
            prediction,
            ground_truth,
            prediction_threshold=0.0,
            prediction_threshold_percentile=97.0,
            ground_truth_threshold=0.0,
            normalisation="clamp",
            metric_mask="ground-truth",
            roi_mask=None,
            ssim_window_size=7,
        )

        self.assertAlmostEqual(threshold, 97.03, places=2)
        self.assertEqual(metrics["prediction_threshold_mode"], "positive-percentile")
        self.assertEqual(metrics["prediction_threshold_domain"], "raw_prediction")
        self.assertEqual(metrics["prediction_foreground_voxels"], 3)
        self.assertAlmostEqual(metrics["masked_dice_3d"], 1.0)
        np.testing.assert_array_equal(
            arrays["prediction_mask_zyx"],
            arrays["ground_truth_mask_zyx"],
        )

    def test_p97_keeps_binary_foreground_above_threshold(self):
        volume = np.zeros((9, 9, 9), dtype=np.float32)
        volume[3:6, 3:6, 3:6] = 1.0
        threshold = positive_percentile_threshold(volume, 97.0)
        self.assertLess(threshold, 1.0)
        self.assertEqual(np.count_nonzero(volume > threshold), 27)

    def test_imagecas_vol_and_spacing_are_loaded_as_physical_xyz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npz"
            np.savez(
                path,
                vol=np.zeros((5, 6, 7), dtype=np.uint8),
                spacing=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            )
            volume, key, axis_order, spacing_xyz_m = load_ground_truth_volume(
                path,
                key=None,
                axis_order="auto",
                spacing_key="spacing",
                spacing_units="mm",
            )
            self.assertEqual(key, "vol")
            self.assertEqual(axis_order, "xyz")
            self.assertEqual(volume.shape, (7, 6, 5))
            np.testing.assert_allclose(
                spacing_xyz_m,
                np.asarray([0.001, 0.002, 0.003]),
            )

    def test_full_gt_is_sampled_at_projection_center_on_prediction_grid(self):
        ground_truth = np.zeros((9, 9, 9), dtype=np.float32)
        ground_truth[2, 2, 2] = 1.0
        aligned = resample_ground_truth_to_prediction_grid(
            ground_truth,
            ground_truth_spacing_xyz_m=(1.0, 1.0, 1.0),
            prediction_shape_zyx=(5, 5, 5),
            prediction_extent_m=4.0,
            projection_center_offset_xyz_m=(2.0, 2.0, 2.0),
            interpolation="nearest",
        )
        self.assertEqual(aligned.shape, (5, 5, 5))
        self.assertEqual(aligned[2, 2, 2], 1.0)
        self.assertEqual(np.count_nonzero(aligned), 1)

    def test_projection_center_offset_is_reversed_in_zyx_order(self):
        centered_prediction = np.zeros((9, 9, 9), dtype=np.float32)
        centered_prediction[4, 4, 4] = 1.0
        # With extent 8 and shape 9, the endpoint-sampled voxel spacing is 1.
        shift_zyx, spacing_zyx = projection_offset_to_voxel_shift_zyx(
            np.asarray([2.0, -1.0, 3.0]),
            centered_prediction.shape,
            volume_extent_m=8.0,
        )
        restored_prediction = translate_volume_zyx(
            centered_prediction,
            shift_zyx,
            interpolation="linear",
        )
        self.assertTrue(np.array_equal(spacing_zyx, np.ones(3)))
        self.assertTrue(np.array_equal(shift_zyx, np.asarray([3.0, -1.0, 2.0])))
        self.assertEqual(restored_prediction[7, 3, 6], 1.0)
        self.assertEqual(np.count_nonzero(restored_prediction), 1)

    def test_fractional_offset_uses_linear_interpolation(self):
        centered_prediction = np.zeros((9, 9, 9), dtype=np.float32)
        centered_prediction[4, 4, 4] = 1.0
        restored_prediction = translate_volume_zyx(
            centered_prediction,
            (0.0, 0.0, 0.5),
            interpolation="linear",
        )
        self.assertAlmostEqual(float(restored_prediction[4, 4, 4]), 0.5)
        self.assertAlmostEqual(float(restored_prediction[4, 4, 5]), 0.5)
        self.assertAlmostEqual(float(restored_prediction.sum()), 1.0)

    def test_identical_volume_has_unit_dice_and_ssim(self):
        volume = np.zeros((9, 9, 9), dtype=np.float32)
        volume[2:7, 3:6, 4:8] = 1.0
        metrics, arrays = compute_volume_metrics(
            volume,
            volume,
            prediction_threshold=0.5,
            ground_truth_threshold=0.5,
            normalisation="clamp",
            metric_mask="ground-truth",
            roi_mask=None,
            ssim_window_size=7,
        )
        self.assertAlmostEqual(metrics["masked_dice_3d"], 1.0)
        self.assertAlmostEqual(metrics["mse_3d"], 0.0)
        self.assertAlmostEqual(metrics["ssim_3d"], 1.0)
        self.assertAlmostEqual(metrics["masked_ssim_3d"], 1.0)
        self.assertEqual(arrays["prediction_mask_zyx"].dtype, np.bool_)

    def test_disjoint_masks_have_zero_dice(self):
        ground_truth = np.zeros((9, 9, 9), dtype=np.float32)
        prediction = np.zeros_like(ground_truth)
        ground_truth[3:5, 3:5, 3:5] = 1.0
        prediction[6:8, 6:8, 6:8] = 1.0
        metrics, _ = compute_volume_metrics(
            prediction,
            ground_truth,
            prediction_threshold=0.5,
            ground_truth_threshold=0.5,
            normalisation="clamp",
            metric_mask="union",
            roi_mask=None,
            ssim_window_size=7,
        )
        self.assertEqual(metrics["masked_dice_3d"], 0.0)
        self.assertGreater(metrics["mse_3d"], 0.0)
        self.assertLess(metrics["ssim_3d"], 1.0)

    def test_ssim_map_is_valid_window_shape(self):
        volume = np.arange(9 ** 3, dtype=np.float32).reshape(9, 9, 9) / float(9 ** 3)
        score, ssim_map = structural_similarity_3d(volume, volume, data_range=1.0, window_size=7)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(ssim_map.shape, (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
