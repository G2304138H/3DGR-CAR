import tempfile
import unittest
from pathlib import Path

import torch

from gcp_losses import (
    GCPLossWeights,
    chamfer_distance,
    gcp_loss,
    soft_cldice_loss,
    trilinear_point_splat,
)
from gcp_model import (
    GCPModelConfig,
    MonocularGaussianCenterPredictor,
    lift_depth_offsets_to_centers,
    load_gcp_checkpoint,
    make_gcp_checkpoint,
)


class GCPModelTests(unittest.TestCase):
    def _model(self) -> MonocularGaussianCenterPredictor:
        return MonocularGaussianCenterPredictor(
            GCPModelConfig(
                image_size=32,
                base_channels=4,
                num_levels=2,
                alpha=2,
                offset_scale=0.075,
                norm_groups=2,
            )
        )

    def test_output_shapes_ranges_and_backward(self):
        torch.manual_seed(2)
        model = self._model()
        image = torch.randn(2, 1, 32, 32, requires_grad=True)
        prediction = model(image)

        self.assertEqual(set(prediction), {"depth", "offsets"})
        self.assertEqual(tuple(prediction["depth"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(prediction["offsets"].shape), (2, 3, 16, 16))
        self.assertTrue(bool((prediction["depth"] >= 0.0).all()))
        self.assertTrue(bool((prediction["depth"] <= 1.0).all()))
        self.assertLessEqual(
            float(prediction["offsets"].abs().max()),
            model.config.offset_scale + 1.0e-6,
        )

        (prediction["depth"].mean() + prediction["offsets"].mean()).backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(bool(torch.isfinite(image.grad).all()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_lifting_supports_mapping_and_separate_tensor_forms(self):
        depth = torch.full((1, 1, 2, 3), 0.25)
        offsets = torch.zeros((1, 3, 2, 3))
        offsets[:, 0] = 0.1
        entry = torch.zeros((1, 2, 3, 3))
        exit = torch.ones((1, 2, 3, 3))

        mapping_centers = lift_depth_offsets_to_centers(
            {"depth": depth, "offsets": offsets}, entry, exit,
        )
        separate_centers = lift_depth_offsets_to_centers(depth, offsets, entry, exit,)
        self.assertEqual(tuple(mapping_centers.shape), (1, 6, 3))
        torch.testing.assert_close(mapping_centers, separate_centers)
        torch.testing.assert_close(
            mapping_centers[0, 0], torch.tensor([0.35, 0.25, 0.25]),
        )
        self.assertTrue(bool((mapping_centers >= 0.0).all()))
        self.assertTrue(bool((mapping_centers <= 1.0).all()))

    def test_checkpoint_roundtrip_preserves_config_and_outputs(self):
        torch.manual_seed(7)
        model = self._model().eval()
        image = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            expected = model(image)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "gcp.pt"
            torch.save(make_gcp_checkpoint(model, epoch=5), checkpoint_path)
            loaded = load_gcp_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(loaded.config, model.config)
        self.assertEqual(loaded.config.downsample_factor, 2)
        self.assertFalse(loaded.training)
        with torch.no_grad():
            actual = loaded(image)
        torch.testing.assert_close(actual["depth"], expected["depth"])
        torch.testing.assert_close(actual["offsets"], expected["offsets"])


class GCPLossTests(unittest.TestCase):
    def test_chunked_chamfer_ignores_nan_padding_and_backpropagates(self):
        predicted = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [float("nan")] * 3]],
            requires_grad=True,
        )
        target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [float("nan")] * 3]])
        target_mask = torch.tensor([[True, True, False]])
        predicted_mask = torch.tensor([[True, True, False]])
        loss = chamfer_distance(
            predicted,
            target,
            target_mask,
            predicted_point_mask=predicted_mask,
            chunk_size=1,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_trilinear_splat_conserves_density_and_has_coordinate_gradient(self):
        points = torch.tensor([[[0.2, 0.4, 0.7]]], requires_grad=True)
        density = trilinear_point_splat(points, (4, 5, 6), saturate=False)
        self.assertEqual(tuple(density.shape), (1, 1, 4, 5, 6))
        torch.testing.assert_close(density.sum(), torch.tensor(1.0))
        self.assertGreater(int(torch.count_nonzero(density)), 1)

        z_coordinates = torch.linspace(0.0, 1.0, 4).reshape(1, 1, 4, 1, 1)
        y_coordinates = torch.linspace(0.0, 1.0, 5).reshape(1, 1, 1, 5, 1)
        x_coordinates = torch.linspace(0.0, 1.0, 6).reshape(1, 1, 1, 1, 6)
        first_moment = (
            density * (z_coordinates + 2.0 * y_coordinates + 3.0 * x_coordinates)
        ).sum()
        first_moment.backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(bool(torch.isfinite(points.grad).all()))
        self.assertGreater(float(points.grad.abs().sum()), 0.0)

    def test_soft_cldice_is_zero_for_identical_volume(self):
        volume = torch.zeros((2, 1, 7, 7, 7))
        volume[:, :, 1:6, 3, 3] = 1.0
        loss = soft_cldice_loss(volume, volume, skeleton_iterations=2)
        self.assertLess(float(loss), 1.0e-5)

    def test_combined_loss_has_all_components_and_gradients(self):
        torch.manual_seed(11)
        predicted_centers = torch.rand((2, 5, 3), requires_grad=True)
        target_points = torch.rand((2, 4, 3))
        target_point_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]]
        )
        predicted_depth = torch.sigmoid(torch.randn((2, 1, 4, 4), requires_grad=True))
        predicted_depth.retain_grad()
        target_depth = torch.full((2, 1, 8, 8), 1.0)
        target_depth[:, :, 2:6, 2:6] = 0.4
        depth_mask = target_depth < 1.0

        volume_logits = torch.randn((2, 1, 7, 7, 7), requires_grad=True)
        predicted_volume = torch.sigmoid(volume_logits)
        target_volume = torch.zeros_like(predicted_volume)
        target_volume[:, :, 1:6, 3, 3] = 1.0
        weights = GCPLossWeights(
            chamfer=1.0, silog=0.5, depth_l1=2.0, depth_gradient=0.25, cldice=0.75,
        )

        total, components = gcp_loss(
            predicted_centers,
            target_points,
            target_point_mask,
            predicted_depth,
            target_depth,
            depth_mask,
            predicted_volume,
            target_volume,
            weights,
            chamfer_chunk_size=2,
            skeleton_iterations=1,
        )
        self.assertEqual(
            set(components),
            {"chamfer", "silog", "depth_l1", "depth_gradient", "cldice"},
        )
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertIsNotNone(predicted_centers.grad)
        self.assertIsNotNone(predicted_depth.grad)
        self.assertIsNotNone(volume_logits.grad)
        self.assertTrue(bool(torch.isfinite(predicted_centers.grad).all()))


if __name__ == "__main__":
    unittest.main()
