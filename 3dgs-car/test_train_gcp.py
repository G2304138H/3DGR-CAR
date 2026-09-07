"""Small CPU tests for GCP training orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from gcp_dataset import gcp_collate
from gcp_losses import GCPLossWeights
from gcp_model import (
    GCPModelConfig,
    MonocularGaussianCenterPredictor,
    make_gcp_checkpoint,
)
from train_gcp import (
    _load_training_checkpoint,
    _make_grad_scaler,
    load_split_case_names,
    parse_args,
    run_epoch,
)


class SplitParsingTests(unittest.TestCase):
    def test_nested_aliases_and_record_identifiers(self):
        document = {
            "dataset": {
                "splits": {
                    "training": [{"case_number": 1}, {"name": "lca_0002"}],
                    "validation": {"cases": ["lca_0003"]},
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_split_case_names(path, "train"), ["1", "lca_0002"])
            self.assertEqual(load_split_case_names(path, "val"), ["lca_0003"])

    def test_feature_paths_are_converted_to_projection_case_names(self):
        document = {
            "train": [
                "/features/vggt/lca/1/prefix_02.npz",
                "/features/vggt/lca/23/prefix_05.npz",
            ],
            "val": ["/features/all_branch/vggt/rca_0366.npz"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_split_case_names(path, "train"), ["lca_0001", "lca_0023"],
            )
            self.assertEqual(load_split_case_names(path, "val"), ["rca_0366"])

    def test_config_supplies_defaults_and_cli_can_override_them(self):
        document = {
            "schema_version": 1,
            "training": {
                "projection_dir": "/projection",
                "ground_truth_dir": "/volume",
                "split_json": "/split.json",
                "output_dir": "/output",
                "expected_detector_pixel_spacing_mm": 0.65,
                "epochs": 100,
                "amp": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            args = parse_args(["--config", str(path), "--epochs", "3", "--no-amp"])
        self.assertEqual(args.projection_dir, "/projection")
        self.assertEqual(args.expected_detector_pixel_spacing_mm, 0.65)
        self.assertEqual(args.epochs, 3)
        self.assertFalse(args.amp)


class _TinyGCPDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(20 + int(index))
        image = torch.rand((1, 16, 16), generator=generator)
        volume = torch.zeros((1, 5, 5, 5), dtype=torch.float32)
        volume[:, 1:4, 2, 2] = 1.0
        depth = torch.full((1, 8, 8), 0.45 + 0.05 * index)
        depth_mask = torch.ones_like(depth, dtype=torch.bool)
        ray_valid = torch.ones((1, 8, 8), dtype=torch.bool)
        ray_valid[:, 0, 0] = False
        ray_entry = torch.zeros((8, 8, 3), dtype=torch.float32)
        ray_exit = torch.ones((8, 8, 3), dtype=torch.float32)
        point_cloud = torch.tensor(
            [[0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]][: 2 + index],
            dtype=torch.float32,
        )
        return {
            "image": image,
            "volume": volume,
            "point_cloud": point_cloud,
            "depth": depth,
            "depth_mask": depth_mask,
            "ray_valid_mask": ray_valid,
            "ray_entry_zyx": ray_entry,
            "ray_exit_zyx": ray_exit,
            "case_name": f"case_{index}",
            "view_index": index,
        }


class TrainingLoopTests(unittest.TestCase):
    def test_one_cpu_optimizer_step_returns_finite_metrics(self):
        torch.manual_seed(4)
        model = MonocularGaussianCenterPredictor(
            GCPModelConfig(
                image_size=16, base_channels=2, num_levels=2, alpha=2, norm_groups=1,
            )
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        loader = DataLoader(
            _TinyGCPDataset(), batch_size=2, shuffle=False, collate_fn=gcp_collate,
        )
        before = model.prediction_head.weight.detach().clone()
        metrics = run_epoch(
            model,
            loader,
            torch.device("cpu"),
            GCPLossWeights(),
            volume_size=5,
            optimizer=optimizer,
            scaler=_make_grad_scaler(False),
            chamfer_chunk_size=16,
            skeleton_iterations=1,
        )

        self.assertEqual(
            set(metrics),
            {"loss", "chamfer", "silog", "depth_l1", "depth_gradient", "cldice"},
        )
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )
        self.assertFalse(torch.equal(before, model.prediction_head.weight.detach()))

    def test_training_checkpoint_loader_roundtrip(self):
        model = MonocularGaussianCenterPredictor(
            GCPModelConfig(image_size=16, base_channels=2, num_levels=2, norm_groups=1)
        )
        optimizer = torch.optim.AdamW(model.parameters())
        generator = torch.Generator().manual_seed(12)
        payload = make_gcp_checkpoint(
            model,
            epoch=3,
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict={},
            data_loader_generator_state=generator.get_state(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            torch.save(payload, path)
            loaded = _load_training_checkpoint(path, torch.device("cpu"))
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["model_config"], model.config.to_dict())
        torch.testing.assert_close(
            loaded["data_loader_generator_state"], generator.get_state(),
        )


if __name__ == "__main__":
    unittest.main()
