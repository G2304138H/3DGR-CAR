#!/usr/bin/env python3
"""Train the monocular Gaussian Centre Predictor on paired Stage-2 NPZ data.

Each projection view is an independent monocular sample.  The trained model
predicts depth and a bounded 3D offset for every cell of a downsampled detector
grid.  Camera rays lift those predictions into normalized ZYX Gaussian centres.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from gcp_dataset import PairedGCPDataset, discover_case_pairs, gcp_collate
from gcp_losses import GCPLossWeights, gcp_loss, trilinear_point_splat
from gcp_model import (
    GCPModelConfig,
    MonocularGaussianCenterPredictor,
    lift_depth_offsets_to_centers,
    make_gcp_checkpoint,
)
from stage2_npz_data import DEFAULT_SOURCE_ORIGIN_DISTANCE_M


def _normalise_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _canonical_case_reference(value: object) -> str:
    """Convert feature/archive paths in split files to projection case names."""

    text = str(value).strip()
    if not text:
        return text
    if isinstance(value, (int, np.integer)) or text.isdigit():
        return str(value)

    parts = [part for part in text.replace("\\", "/").split("/") if part]
    stem = parts[-1].rsplit(".", 1)[0] if parts else text
    direct = re.fullmatch(r"(?i)(lca|rca)[_-]?0*([0-9]+)", stem)
    if direct is not None:
        vessel, case_number = direct.groups()
        return f"{vessel.lower()}_{int(case_number):04d}"

    # LCA split entries point to files such as ``lca/1/prefix_02.npz``.
    # Recover the physical case from the vessel directory and numeric child.
    for index, part in enumerate(parts[:-1]):
        vessel = part.lower()
        if vessel not in {"lca", "rca"}:
            continue
        for child in parts[index + 1 : -1]:
            if child.isdigit():
                return f"{vessel}_{int(child):04d}"

    return text


def _find_split(document: object, requested: str) -> object:
    aliases = {
        "train": {"train", "training"},
        "val": {"val", "validation", "valid", "dev"},
        "test": {"test", "testing"},
    }
    requested_key = _normalise_key(requested)
    canonical = next(
        (name for name, names in aliases.items() if requested_key in names),
        requested_key,
    )

    def search(value: object) -> Optional[object]:
        if not isinstance(value, Mapping):
            return None
        for key, child in value.items():
            key_normalised = _normalise_key(key)
            if key_normalised == canonical or key_normalised in aliases.get(
                canonical, set()
            ):
                return child
        for container in ("splits", "partitions", "dataset"):
            for key, child in value.items():
                if _normalise_key(key) == container:
                    result = search(child)
                    if result is not None:
                        return result
        return None

    found = search(document)
    if found is None:
        raise KeyError(f"Split {requested!r} was not found in the split document.")
    return found


def _case_reference(record: object) -> str:
    if isinstance(record, (str, int, np.integer)):
        return _canonical_case_reference(record)
    if not isinstance(record, Mapping):
        raise TypeError(f"Unsupported split record: {record!r}.")
    for candidate in (
        "path",
        "file",
        "source_path",
        "case_name",
        "sample_name",
        "case_id",
        "case_number",
        "case",
        "id",
        "name",
    ):
        for key, value in record.items():
            if _normalise_key(key) == _normalise_key(candidate):
                return _canonical_case_reference(value)
    raise ValueError(f"Split record has no recognized case identifier: {record!r}.")


def load_split_case_names(path: Path, split: str) -> List[str]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    value = _find_split(document, split)
    if isinstance(value, Mapping):
        for candidate in ("cases", "case_names", "case_ids", "items", "samples"):
            match = next(
                (
                    child
                    for key, child in value.items()
                    if _normalise_key(key) == _normalise_key(candidate)
                ),
                None,
            )
            if match is not None:
                value = match
                break
    if not isinstance(value, list):
        raise TypeError(
            f"Split {split!r} must resolve to a list, got {type(value).__name__}."
        )
    names = [_case_reference(record).strip() for record in value]
    if not names or any(not name for name in names):
        raise ValueError(
            f"Split {split!r} is empty or contains an empty case identifier."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Split {split!r} contains duplicate case identifiers.")
    return names


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def _finite_loss(loss: torch.Tensor, batch: Mapping[str, Any]) -> None:
    if not bool(torch.isfinite(loss)):
        cases = batch.get("case_name", "unknown")
        views = batch.get("view_index", "unknown")
        raise FloatingPointError(
            f"Non-finite GCP loss for cases={cases}, views={views}."
        )


def _make_grad_scaler(enabled: bool) -> Any:
    """Create a CUDA gradient scaler across old and new PyTorch AMP APIs."""

    amp_module = getattr(torch, "amp", None)
    scaler_class = getattr(amp_module, "GradScaler", None)
    if scaler_class is not None:
        try:
            return scaler_class("cuda", enabled=enabled)
        except TypeError:
            return scaler_class(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool, device: torch.device) -> Any:
    if not enabled:
        return nullcontext()
    amp_module = getattr(torch, "amp", None)
    autocast = getattr(amp_module, "autocast", None)
    if autocast is not None:
        try:
            return autocast(device_type=device.type, enabled=True)
        except TypeError:
            return autocast(device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _load_training_checkpoint(
    path: str | Path, device: torch.device
) -> Mapping[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Resume checkpoint {checkpoint_path} must contain a mapping.")
    required = {"model_config", "model_state_dict", "optimizer_state_dict", "epoch"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(
            f"Resume checkpoint {checkpoint_path} is missing keys: {missing}."
        )
    return checkpoint


def run_epoch(
    model: MonocularGaussianCenterPredictor,
    loader: DataLoader,
    device: torch.device,
    loss_weights: GCPLossWeights,
    volume_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    gradient_clip_norm: float = 0.0,
    chamfer_chunk_size: int = 1024,
    skeleton_iterations: int = 3,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    sample_count = 0
    amp_enabled = bool(scaler is not None and scaler.is_enabled())

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size = int(batch["image"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(amp_enabled, device):
                prediction = model(batch["image"])
                # CUDA autocast may emit half-precision predictions while the
                # ray geometry and supervision tensors remain float32. Lift
                # and supervise in the target dtype; these casts retain the
                # gradient path back into the autocast model.
                prediction_for_loss = {
                    "depth": prediction["depth"].to(dtype=batch["depth"].dtype),
                    "offsets": prediction["offsets"].to(
                        dtype=batch["ray_entry_zyx"].dtype
                    ),
                }
                predicted_centers = lift_depth_offsets_to_centers(
                    prediction_for_loss, batch["ray_entry_zyx"], batch["ray_exit_zyx"],
                )
                predicted_point_mask = (
                    batch["ray_valid_mask"]
                    .reshape(batch_size, -1,)
                    .to(dtype=torch.bool)
                )
                if predicted_point_mask.shape != predicted_centers.shape[:2]:
                    raise ValueError(
                        "ray_valid_mask and lifted centres disagree: "
                        f"{tuple(predicted_point_mask.shape)} versus "
                        f"{tuple(predicted_centers.shape[:2])}."
                    )
                predicted_volume = trilinear_point_splat(
                    predicted_centers,
                    grid_size=int(volume_size),
                    point_mask=predicted_point_mask,
                )
                loss, components = gcp_loss(
                    predicted_centers,
                    batch["point_cloud"],
                    batch["point_mask"],
                    prediction_for_loss["depth"],
                    batch["depth"],
                    batch["depth_mask"],
                    predicted_volume,
                    batch["volume"],
                    loss_weights,
                    predicted_point_mask=predicted_point_mask,
                    chamfer_chunk_size=int(chamfer_chunk_size),
                    skeleton_iterations=int(skeleton_iterations),
                )
                _finite_loss(loss, batch)

            if training:
                if amp_enabled:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if gradient_clip_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip_norm
                        )
                    optimizer.step()

        values = {"loss": loss.detach(), **components}
        for name, value in values.items():
            scalar = float(value.detach().item() if torch.is_tensor(value) else value)
            totals[name] = totals.get(name, 0.0) + scalar * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The GCP data loader produced no samples.")
    return {name: value / sample_count for name, value in totals.items()}


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_training_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, Mapping):
        raise TypeError("The GCP training config must contain a JSON object.")
    unexpected_top_level = sorted(
        set(document).difference({"schema_version", "description", "training"})
    )
    if unexpected_top_level:
        raise ValueError(
            "Unexpected top-level GCP config keys: "
            f"{unexpected_top_level}. Put CLI defaults under 'training'."
        )
    if document.get("schema_version", 1) != 1:
        raise ValueError("Only GCP training config schema_version 1 is supported.")
    training = document.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("The GCP training config requires a 'training' object.")
    return {
        str(key).strip().replace("-", "_"): value for key, value in training.items()
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = (
        {} if config_args.config is None else _load_training_config(config_args.config)
    )

    parser = argparse.ArgumentParser(
        description="Train the monocular Gaussian Centre Predictor."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON file containing defaults under a 'training' object.",
    )
    parser.add_argument("--projection-dir", default=None)
    parser.add_argument("--ground-truth-dir", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--volume-extent-m", type=float, default=None)
    parser.add_argument(
        "--source-origin-distance-m",
        type=float,
        default=DEFAULT_SOURCE_ORIGIN_DISTANCE_M,
        help="Source-to-isocentre distance used to construct cone-beam rays.",
    )
    parser.add_argument(
        "--expected-detector-pixel-spacing-mm",
        type=float,
        default=None,
        help=(
            "Optional data-integrity check. Geometry always uses the spacing stored "
            "in each projection NPZ."
        ),
    )
    parser.add_argument(
        "--fallback-detector-pixel-spacing-mm",
        type=float,
        default=None,
        help=(
            "Detector pixel spacing in mm used only when a projection NPZ has no "
            "imager_pixel_spacing key."
        ),
    )
    parser.add_argument(
        "--fallback-sid-m",
        type=float,
        default=None,
        help="SID in metres used only when a projection NPZ has no sid key.",
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--offset-scale", type=float, default=0.1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--norm-groups", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--chamfer-weight", type=float, default=1.0)
    parser.add_argument("--silog-weight", type=float, default=0.01)
    parser.add_argument("--depth-l1-weight", type=float, default=0.01)
    parser.add_argument("--depth-gradient-weight", type=float, default=0.01)
    parser.add_argument("--cldice-weight", type=float, default=0.5)
    parser.add_argument("--chamfer-chunk-size", type=int, default=1024)
    parser.add_argument("--skeleton-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--resume", default=None)
    allowed_config_keys = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unexpected_config_keys = sorted(
        set(config_defaults).difference(allowed_config_keys)
    )
    if unexpected_config_keys:
        parser.error(f"Unexpected GCP training settings: {unexpected_config_keys}.")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    missing_paths = [
        option
        for option, value in (
            ("--projection-dir", args.projection_dir),
            ("--ground-truth-dir", args.ground_truth_dir),
            ("--split-json", args.split_json),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing_paths:
        parser.error(
            f"Missing required settings: {missing_paths}. Supply them by CLI or --config."
        )
    if args.image_size <= 0 or args.volume_size <= 1:
        parser.error("--image-size must be positive and --volume-size must exceed one.")
    if args.downsample_factor <= 0 or args.image_size % args.downsample_factor != 0:
        parser.error("--downsample-factor must divide --image-size exactly.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    if args.max_points is not None and args.max_points <= 0:
        parser.error("--max-points must be positive.")
    if args.source_origin_distance_m <= 0.0:
        parser.error("--source-origin-distance-m must be positive.")
    if args.expected_detector_pixel_spacing_mm is not None and (
        not math.isfinite(float(args.expected_detector_pixel_spacing_mm))
        or float(args.expected_detector_pixel_spacing_mm) <= 0.0
    ):
        parser.error(
            "--expected-detector-pixel-spacing-mm must be finite and positive."
        )
    for option, value in (
        (
            "--fallback-detector-pixel-spacing-mm",
            args.fallback_detector_pixel_spacing_mm,
        ),
        ("--fallback-sid-m", args.fallback_sid_m),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            parser.error(f"{option} must be finite and positive.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    requested_device = torch.device(str(args.device))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = requested_device

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else output_dir / "target_cache"
    )
    split_path = Path(args.split_json).expanduser().resolve()
    train_names = load_split_case_names(split_path, str(args.train_split))
    validation_names = load_split_case_names(split_path, str(args.validation_split))
    overlap = sorted(set(train_names).intersection(validation_names))
    if overlap:
        raise ValueError(f"Train/validation leakage: {overlap[:10]}.")

    train_pairs = discover_case_pairs(
        args.projection_dir,
        args.ground_truth_dir,
        train_names,
        expected_detector_pixel_spacing_mm=args.expected_detector_pixel_spacing_mm,
    )
    validation_pairs = discover_case_pairs(
        args.projection_dir,
        args.ground_truth_dir,
        validation_names,
        expected_detector_pixel_spacing_mm=args.expected_detector_pixel_spacing_mm,
    )
    resolved_overlap = sorted(
        {pair.projection_path for pair in train_pairs}.intersection(
            pair.projection_path for pair in validation_pairs
        )
    )
    if resolved_overlap:
        raise ValueError(
            "Train/validation leakage after resolving case aliases: "
            f"{[str(path) for path in resolved_overlap[:10]]}."
        )
    dataset_kwargs = dict(
        volume_size=int(args.volume_size),
        image_size=int(args.image_size),
        volume_extent_m=args.volume_extent_m,
        downsample_factor=int(args.downsample_factor),
        cache_dir=cache_dir,
        max_points=int(args.max_points),
        source_origin_distance_m=float(args.source_origin_distance_m),
        fallback_detector_pixel_spacing_mm=(
            args.fallback_detector_pixel_spacing_mm
        ),
        fallback_sid_m=args.fallback_sid_m,
    )
    train_dataset = PairedGCPDataset(train_pairs, **dataset_kwargs)
    validation_dataset = PairedGCPDataset(validation_pairs, **dataset_kwargs)

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader_kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        collate_fn=gcp_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model_config = GCPModelConfig(
        image_size=int(args.image_size),
        base_channels=int(args.base_channels),
        num_levels=int(args.num_levels),
        alpha=int(args.downsample_factor),
        offset_scale=float(args.offset_scale),
        norm_groups=int(args.norm_groups),
        dropout=float(args.dropout),
    )
    model = MonocularGaussianCenterPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = _make_grad_scaler(enabled=bool(args.amp and device.type == "cuda"))
    loss_weights = GCPLossWeights(
        chamfer=float(args.chamfer_weight),
        silog=float(args.silog_weight),
        depth_l1=float(args.depth_l1_weight),
        depth_gradient=float(args.depth_gradient_weight),
        cldice=float(args.cldice_weight),
    )

    start_epoch = 0
    best_validation_loss = math.inf
    if args.resume is not None:
        checkpoint = _load_training_checkpoint(args.resume, device)
        checkpoint_config = GCPModelConfig.from_dict(checkpoint["model_config"])
        if checkpoint_config != model_config:
            raise ValueError(
                "Resume checkpoint model configuration does not match the CLI."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler.is_enabled() and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("data_loader_generator_state") is not None:
            generator.set_state(checkpoint["data_loader_generator_state"].cpu())
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))

    configuration = {
        **vars(args),
        "projection_dir": str(Path(args.projection_dir).expanduser().resolve()),
        "ground_truth_dir": str(Path(args.ground_truth_dir).expanduser().resolve()),
        "split_json": str(split_path),
        "config": (
            None
            if args.config is None
            else str(Path(args.config).expanduser().resolve())
        ),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "model_config": model_config.to_dict(),
        "loss_weights": asdict(loss_weights),
        "train_cases": [pair.case_name for pair in train_pairs],
        "validation_cases": [pair.case_name for pair in validation_pairs],
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "coordinate_order": "normalized_zyx",
        "depth_background": 1.0,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            loss_weights,
            int(args.volume_size),
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(args.gradient_clip_norm),
            chamfer_chunk_size=int(args.chamfer_chunk_size),
            skeleton_iterations=int(args.skeleton_iterations),
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                loss_weights,
                int(args.volume_size),
                chamfer_chunk_size=int(args.chamfer_chunk_size),
                skeleton_iterations=int(args.skeleton_iterations),
            )
        record = {
            "epoch": epoch,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint = make_gcp_checkpoint(
            model,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            data_loader_generator_state=generator.get_state(),
            best_validation_loss=min(best_validation_loss, validation_metrics["loss"]),
            training_config=configuration,
            metrics=record,
        )
        _atomic_torch_save(checkpoint, output_dir / "last_gcp.pt")
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            _atomic_torch_save(checkpoint, output_dir / "best_gcp.pt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
