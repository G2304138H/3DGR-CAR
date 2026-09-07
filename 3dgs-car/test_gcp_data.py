"""CPU-only tests for GCP pairing, alignment, rays, and collation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gcp_dataset import (
    GCPCasePair,
    PairedGCPDataset,
    discover_case_pairs,
    gcp_collate,
    torch,
)
from gcp_targets import (
    detector_ray_box_intersections,
    load_ground_truth_npz,
    raycast_first_hit_nearest,
    resample_volume_to_centered_cube,
    scale_cone_vector_for_detector,
    volume_to_normalized_points,
)


def _write_projection(
    path: Path, sample_name: str, case_id: str, views: int = 2
) -> None:
    np.savez(
        path,
        sample_name=np.asarray(sample_name),
        vessel_type=np.asarray("lca"),
        case_id=np.asarray(case_id),
        images=np.zeros((views, 8, 8), dtype=np.float32),
        theta_deg=np.zeros(views, dtype=np.float32),
        phi_deg=np.full(views, 90.0, dtype=np.float32),
        sid=np.asarray(0.9, dtype=np.float32),
        imager_pixel_spacing=np.asarray(10.0, dtype=np.float32),
        imager_pixel_spacing_units=np.asarray("mm"),
        projection_center_offset=np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
        clinical_views=np.asarray([f"view_{index}" for index in range(views)]),
    )


class GroundTruthAlignmentTests(unittest.TestCase):
    def test_imagecas_vol_is_loaded_xyz_to_zyx(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1.npz"
            volume_xyz = np.zeros((5, 6, 7), dtype=np.uint8)
            volume_xyz[2, 3, 4] = 1
            np.savez(path, vol=volume_xyz, spacing=np.asarray([1.0, 2.0, 3.0]))

            loaded = load_ground_truth_npz(path)

            self.assertEqual(loaded.volume_key, "vol")
            self.assertEqual(loaded.source_axis_order, "xyz")
            self.assertEqual(loaded.volume_zyx.shape, (7, 6, 5))
            self.assertEqual(loaded.volume_zyx[4, 3, 2], 1.0)
            np.testing.assert_allclose(loaded.spacing_xyz_m, [0.001, 0.002, 0.003])

    def test_physical_projection_center_maps_to_output_center(self) -> None:
        volume_zyx = np.zeros((7, 6, 5), dtype=np.float32)
        volume_zyx[4, 3, 2] = 1.0

        aligned = resample_volume_to_centered_cube(
            volume_zyx=volume_zyx,
            spacing_xyz_m=(1.0, 1.0, 1.0),
            output_shape_zyx=(5, 5, 5),
            volume_extent_m=4.0,
            projection_center_offset_xyz_m=(2.0, 3.0, 4.0),
        )

        self.assertEqual(aligned[2, 2, 2], 1.0)
        self.assertEqual(np.count_nonzero(aligned), 1)

    def test_foreground_points_are_normalized_in_zyx_order(self) -> None:
        volume_zyx = np.zeros((7, 6, 5), dtype=np.float32)
        volume_zyx[4, 3, 2] = 1.0

        points = volume_to_normalized_points(volume_zyx)

        np.testing.assert_allclose(points, [[4.0 / 6.0, 3.0 / 5.0, 2.0 / 4.0]])


class RayTargetTests(unittest.TestCase):
    @staticmethod
    def _axial_cone_vector() -> np.ndarray:
        # Source and detector lie on the Z axis. U advances X columns and V
        # advances Y rows, following ASTRA cone_vec conventions.
        return np.asarray(
            [
                0.0,
                0.0,
                -2.0,
                0.0,
                0.0,
                2.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def test_center_ray_enters_and_exits_expected_cube_faces(self) -> None:
        entry, exit, valid = detector_ray_box_intersections(
            self._axial_cone_vector(), detector_shape=(1, 1), volume_extent_m=2.0
        )

        self.assertTrue(valid[0, 0])
        np.testing.assert_allclose(entry[0, 0], [0.0, 0.5, 0.5], atol=1e-7)
        np.testing.assert_allclose(exit[0, 0], [1.0, 0.5, 0.5], atol=1e-7)

    def test_detector_u_is_column_and_v_is_row(self) -> None:
        entry, _, valid = detector_ray_box_intersections(
            self._axial_cone_vector(), detector_shape=(3, 3), volume_extent_m=2.0
        )

        self.assertTrue(valid.all())
        self.assertGreater(entry[1, 2, 2], entry[1, 0, 2])  # right column -> +X
        self.assertGreater(entry[2, 1, 1], entry[0, 1, 1])  # lower row -> +Y
        self.assertAlmostEqual(float(entry[1, 1, 0]), 0.0)

    def test_detector_resize_preserves_extent_and_axes(self) -> None:
        original = self._axial_cone_vector()
        scaled = scale_cone_vector_for_detector(
            original, original_shape=(8, 12), target_shape=(4, 3)
        )
        np.testing.assert_allclose(scaled[6:9], original[6:9] * 4.0)
        np.testing.assert_allclose(scaled[9:12], original[9:12] * 2.0)

    def test_nearest_raycast_returns_first_hit_fraction_and_background_one(
        self,
    ) -> None:
        volume = np.zeros((5, 5, 5), dtype=np.float32)
        volume[2, 2, 2] = 1.0
        entry = np.asarray([[[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]]], dtype=np.float32)
        exit = np.asarray([[[1.0, 0.5, 0.5], [1.0, 0.0, 0.0]]], dtype=np.float32)

        depth, mask = raycast_first_hit_nearest(volume, entry, exit, num_samples=5)

        self.assertTrue(mask[0, 0])
        self.assertAlmostEqual(float(depth[0, 0]), 0.5)
        self.assertFalse(mask[0, 1])
        self.assertAlmostEqual(float(depth[0, 1]), 1.0)


class PairAndDatasetTests(unittest.TestCase):
    def test_pair_discovery_prefers_vessel_case_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection_dir = root / "projections"
            ground_truth_dir = root / "ground_truth"
            (ground_truth_dir / "lca").mkdir(parents=True)
            projection_dir.mkdir()
            projection_path = projection_dir / "anything.npz"
            expected_gt = ground_truth_dir / "lca" / "1.npz"
            _write_projection(projection_path, "lca_0001", "1")
            np.savez(
                expected_gt,
                vol=np.zeros((3, 3, 3), dtype=np.uint8),
                spacing=np.ones(3, dtype=np.float32),
            )

            pairs = discover_case_pairs(
                projection_dir, ground_truth_dir, case_names=["lca_0001"]
            )

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].projection_path, projection_path.resolve())
            self.assertEqual(pairs[0].ground_truth_path, expected_gt.resolve())
            self.assertEqual(pairs[0].case_name, "lca_0001")

    def test_dataset_indexes_every_view_and_can_use_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection_path = root / "lca_0001.npz"
            ground_truth_path = root / "1.npz"
            cache_dir = root / "cache"
            _write_projection(projection_path, "lca_0001", "1", views=2)
            volume_xyz = np.zeros((5, 5, 5), dtype=np.uint8)
            volume_xyz[2, 2, 2] = 1
            np.savez(
                ground_truth_path,
                vol=volume_xyz,
                spacing=np.full(3, 100.0, dtype=np.float32),
            )
            pair = GCPCasePair(
                projection_path,
                ground_truth_path,
                case_name="lca_0001",
                vessel_type="lca",
                case_id="1",
            )
            dataset = PairedGCPDataset(
                [pair],
                volume_size=5,
                image_size=4,
                volume_extent_m=0.4,
                downsample_factor=2,
                cache_dir=cache_dir,
            )

            first = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(first["image"].shape), (1, 4, 4))
            self.assertEqual(tuple(first["volume"].shape), (1, 5, 5, 5))
            self.assertEqual(tuple(first["depth"].shape), (1, 2, 2))
            self.assertEqual(tuple(first["ray_entry_zyx"].shape), (2, 2, 3))
            self.assertEqual(tuple(first["point_cloud"].shape), (1, 3))
            self.assertEqual(first["view_index"], 0)
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 1)

    def test_collate_pads_variable_point_clouds(self) -> None:
        if torch is None:
            first_points = np.ones((2, 3), dtype=np.float32)
            second_points = np.full((4, 3), 2.0, dtype=np.float32)
            first_image = np.zeros((1, 2, 2), dtype=np.float32)
            second_image = np.ones((1, 2, 2), dtype=np.float32)
        else:
            first_points = torch.ones((2, 3), dtype=torch.float32)
            second_points = torch.full((4, 3), 2.0, dtype=torch.float32)
            first_image = torch.zeros((1, 2, 2), dtype=torch.float32)
            second_image = torch.ones((1, 2, 2), dtype=torch.float32)
        batch = [
            {
                "point_cloud": first_points,
                "image": first_image,
                "case_name": "a",
                "view_index": 0,
            },
            {
                "point_cloud": second_points,
                "image": second_image,
                "case_name": "b",
                "view_index": 1,
            },
        ]

        collated = gcp_collate(batch)

        self.assertEqual(tuple(collated["point_cloud"].shape), (2, 4, 3))
        self.assertEqual(tuple(collated["point_mask"].shape), (2, 4))
        np.testing.assert_array_equal(
            np.asarray(collated["point_mask"]),
            [[True, True, False, False], [True, True, True, True]],
        )
        self.assertEqual(collated["case_name"], ["a", "b"])
        self.assertEqual(collated["view_index"], [0, 1])


if __name__ == "__main__":
    unittest.main()
