import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_stage2_npz import (
    NpzIndex,
    compute_volume_metrics,
    load_ground_truth_volume,
    load_split_case_references,
    parse_args,
    projection_offset_to_voxel_shift_zyx,
    resample_ground_truth_to_prediction_grid,
    structural_similarity_3d,
    translate_volume_zyx,
)


class SplitLoadingTests(unittest.TestCase):
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

    def test_split_key_with_case_numbers_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"test_case_numbers": [3, 8]}), encoding="utf-8")
            self.assertEqual(load_split_case_references(path, "test"), ["3", "8"])


class MetricTests(unittest.TestCase):
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
