import json
import unittest

import torch
import torch.nn.functional as F

from gaussian_model_anisotropic import GaussianModelAnisotropic
from train_stage2_npz import (
    build_centerline_skeleton_masks,
    parse_args,
    projection_reconstruction_loss,
    resolve_projection_loss_alpha,
    select_gcp_initialization_view,
)


class Stage2GCPHelperTests(unittest.TestCase):
    def test_projection_calibration_fallback_arguments_are_parsed(self):
        args, _ = parse_args(
            [
                "--input",
                "case.npz",
                "--output-dir",
                "output",
                "--fallback-detector-pixel-spacing-mm",
                "0.55",
                "--fallback-sid-m",
                "0.9",
            ]
        )
        self.assertEqual(args.fallback_detector_pixel_spacing_mm, 0.55)
        self.assertEqual(args.fallback_sid_m, 0.9)

    def test_expected_gcp_model_parameters_are_parsed(self):
        expected = {
            "image_size": 128,
            "in_channels": 1,
            "base_channels": 32,
            "num_levels": 4,
            "alpha": 2,
            "offset_scale": 0.1,
            "norm_groups": 8,
            "dropout": 0.0,
        }
        args, _ = parse_args(
            [
                "--input",
                "case.npz",
                "--output-dir",
                "output",
                "--init-method",
                "gcp",
                "--gcp-checkpoint",
                "best_gcp.pt",
                "--expected-gcp-model-config-json",
                json.dumps(expected),
            ]
        )
        self.assertEqual(args.expected_gcp_model_config, expected)

    def test_fixed_view_direction_arguments_are_parsed(self):
        args, _ = parse_args(
            [
                "--input",
                "case.npz",
                "--output-dir",
                "output",
                "--view-direction-theta-change-deg",
                "5",
                "--view-direction-phi-change-deg",
                "-2.5",
            ]
        )
        self.assertEqual(args.view_direction_theta_change_deg, 5.0)
        self.assertEqual(args.view_direction_phi_change_deg, -2.5)

    def test_monocular_initializer_selects_exactly_first_requested_view(self):
        self.assertEqual(select_gcp_initialization_view([3, 5]), 3)
        self.assertEqual(select_gcp_initialization_view([6, 0, 4]), 6)
        with self.assertRaises(ValueError):
            select_gcp_initialization_view([])

    def test_projection_loss_defaults_preserve_legacy_baselines(self):
        self.assertEqual(resolve_projection_loss_alpha("fdk", None), 1.0)
        self.assertEqual(resolve_projection_loss_alpha("bp", None), 1.0)
        self.assertEqual(resolve_projection_loss_alpha("gcp", None), 0.5)
        self.assertEqual(resolve_projection_loss_alpha("gcp", 0.75), 0.75)

    def test_projection_loss_arithmetic_and_gradient(self):
        prediction = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        centerline = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

        loss, image_mse, centerline_mse = projection_reconstruction_loss(
            prediction, target, alpha=0.25, centerline_masks=centerline,
        )
        self.assertAlmostEqual(float(image_mse), 3.5)
        self.assertAlmostEqual(float(centerline_mse), 2.0)
        self.assertAlmostEqual(float(loss), 2.375)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_alpha_one_is_exact_plain_mse_and_needs_no_skeleton(self):
        prediction = torch.tensor([[[[0.2, 0.8]]]])
        target = torch.tensor([[[[0.0, 1.0]]]])
        loss, image_mse, _ = projection_reconstruction_loss(
            prediction, target, alpha=1.0
        )
        self.assertTrue(torch.equal(loss, F.mse_loss(prediction, target)))
        self.assertTrue(torch.equal(loss, image_mse))

    def test_binary_target_skeletons_are_built_once_per_view_shape(self):
        target = torch.zeros((1, 2, 7, 7), dtype=torch.float32)
        target[0, 0, 2:5, 2:5] = 1.0
        target[0, 1, 3, 1:6] = 1.0
        skeleton = build_centerline_skeleton_masks(target)
        self.assertEqual(skeleton.shape, target.shape)
        self.assertEqual(skeleton.dtype, target.dtype)
        self.assertGreater(float(skeleton.sum()), 0.0)
        self.assertTrue(torch.all(skeleton <= target))


class PredictedCenterInitializationTests(unittest.TestCase):
    def test_initializer_uses_model_activation_parameterizations_on_cpu(self):
        centers = torch.tensor(
            [[-0.1, 0.25, 0.50], [0.75, 1.1, 0.25]], dtype=torch.float32
        )
        gaussians = GaussianModelAnisotropic()
        gaussians.create_from_predicted_centers(
            centers,
            ini_density=0.2,
            ini_sigma=0.05,
            scale_from_nearest_neighbour=False,
        )

        self.assertEqual(gaussians.get_gaussians_num, 2)
        self.assertTrue(torch.all(gaussians.get_xyz >= 0.0))
        self.assertTrue(torch.all(gaussians.get_xyz <= 1.0))
        torch.testing.assert_close(
            gaussians.get_density, torch.full((2, 1), 0.2),
        )
        torch.testing.assert_close(
            gaussians.get_scaling, torch.full((2, 3), 0.05),
        )
        torch.testing.assert_close(
            gaussians.get_rotation, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        )

    def test_initializer_discards_nonfinite_centers(self):
        centers = torch.tensor(
            [[0.1, 0.2, 0.3], [float("nan"), 0.0, 0.0]], dtype=torch.float32,
        )
        gaussians = GaussianModelAnisotropic()
        gaussians.create_from_predicted_centers(
            centers, scale_from_nearest_neighbour=False,
        )
        self.assertEqual(gaussians.get_gaussians_num, 1)


if __name__ == "__main__":
    unittest.main()
